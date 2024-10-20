const layer = @import("layer.zig");
const layerB = @import("layerBias.zig");
const layerG = @import("layerGrok.zig");
const nll = @import("nll.zig");
const diem = @import("diem.zig");
const dataSet = @import("dataSet.zig");
const relu = @import("relu.zig");
const reloid = @import("reloid.zig");
const pyramid = @import("pyramid.zig");
const gaussian = @import("gaussian.zig");
const pGaussian = @import("parGaussian.zig");

const utils = @import("utils.zig");
const std = @import("std");

const timer = false;

const scheduleItem = struct {
    epochs: usize,
    hLSize: usize,
};
const schedule = [_]scheduleItem{
    .{ .epochs = 25, .hLSize = 10 },
    .{ .epochs = 10, .hLSize = 25 },
    .{ .epochs = 5, .hLSize = 40 },
    .{ .epochs = 5, .hLSize = 100 },
};

const continueFrom = 0;
//todo perlayer configs
const l2_lambda = 0.0075;
const m = std.math;
const regDim: f64 = m.phi;

const graphfuncs = false;
const multiIter = true;

const fileSignature = "G25RRRR_G10R.f64";

const reinit = false;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};

    const allocator = gpa.allocator();

    const dataset = try dataSet.mnist.dtype.readData(allocator);
    defer dataset.deinit(allocator);

    if (graphfuncs) {
        utils.graphFunc(gaussian);
    }

    comptime var itera: usize = continueFrom;

    std.debug.print("Training... \n", .{});

    const t = std.time.milliTimestamp();
    inline while (itera < schedule.len and multiIter) : (itera += 1) {
        var arena = std.heap.ArenaAllocator.init(allocator);
        try runSchedule(itera, dataset, arena.allocator());
        arena.deinit();
    }

    const ct = std.time.milliTimestamp();

    std.debug.print("time total: {}\n", .{ct - t});
}

fn initLayers(comptime layers: []const uLayer, config: anytype, dataset: anytype, allocator: std.mem.Allocator) ![layers.len]Layer {
    comptime var previousLayerSize = dataset.inputSize;
    var storage: [layers.len]Layer = undefined;
    inline for (layers, 0..) |lay, i| {
        storage[i] = try layerInit(
            allocator,
            lay,
            .{
                .batchSize = config.batchSize,
                .inputSize = previousLayerSize,
            },
        );
        if (config.deinitbkw) {
            switch (storage[i]) {
                inline else => |*l| l.deinitBackwards(allocator),
            }
        }
        switch (lay) {
            .Layer, .LayerB, .LayerG => |size| previousLayerSize = size,
            .PGaussian => previousLayerSize = previousLayerSize - 3,
            else => {},
        }
    }
    return storage;
}

fn runSchedule(comptime itera: usize, dataset: anytype, allocator: std.mem.Allocator) !void {
    const first = itera == 0;

    const resize = !first;
    const readfile = !first;
    const writeFile = true;

    const batchSize = 100;
    const epochs = schedule[itera].epochs;
    const ps = if (!first) schedule[itera - 1].hLSize else 0;
    const cs = schedule[itera].hLSize;

    var sum: usize = 0;
    for (schedule[0..itera]) |elem| {
        sum += elem.epochs;
    }
    const pastEp = sum;
    //std.debug.print("last:{}", .{primes100[primes100.len - 1]});
    //var asd = try @import("clKernel.zig").init();
    //try asd.run();

    const file = try std.fs.cwd().createFile(
        "data/Params_" ++ fileSignature,
        .{
            .read = true,
            .truncate = false,
        },
    );
    const default = uLayer.Relu;

    const fileL = [_]uLayer{
        .{ .LayerG = ps }, default,
        .{ .LayerG = ps }, .Reloid,
        .{ .LayerG = ps }, .Reloid,
        .{ .LayerG = ps }, .Reloid,
        .{ .LayerG = 10 }, default,
    };
    comptime var previousLayerSizeF = dataset.inputSize;
    //TODO: test PGaussian
    const layers = comptime [_]uLayer{
        .{ .LayerG = cs }, default,
        .{ .LayerG = cs }, .Reloid,
        .{ .LayerG = cs }, .Reloid,
        .{ .LayerG = cs }, .Reloid,
        .{ .LayerG = 10 }, default,
    };

    var storage = try initLayers(
        &layers,
        .{ .deinitbkw = false, .batchSize = batchSize },
        dataset,
        allocator,
    );
    var validationStorage = try initLayers(
        &layers,
        .{ .deinitbkw = true, .batchSize = dataset.validationSize },
        dataset,
        allocator,
    );

    var reader = std.io.bufferedReader(file.reader());

    if (readfile) {
        inline for (fileL, 0..) |lay, i| {
            if (resize) {
                var other = try layerInit(
                    allocator,
                    lay,
                    .{
                        .batchSize = batchSize,
                        .inputSize = previousLayerSizeF,
                    },
                );
                try utils.callIfCanErr(&other, &reader, "readParams");
                utils.callIfTypeMatch(&storage[i], other, "rescale");
                switch (lay) {
                    .Layer, .LayerB, .LayerG => |size| previousLayerSizeF = size,
                    .PGaussian => previousLayerSizeF = previousLayerSizeF - 3,
                    else => {},
                }
            } else {
                try utils.callIfCanErr(&storage[i], &reader, "readParams");
            }
            if (reinit) {
                try utils.callIfCanErr(&storage[i], 1.0, "reinit");
            }
        }
    }

    file.close();

    const k = try Neuralnet(
        &validationStorage,
        &storage,
        dataset,
        .{ .epochs = epochs, .batchSize = batchSize, .from = pastEp },
        allocator,
    );

    if (writeFile) {
        const filew = try std.fs.cwd().createFile(
            "data/Params_" ++ fileSignature,
            .{
                .read = true,
                .truncate = true,
            },
        );
        defer filew.close();
        for (0..k.len) |l| {
            try utils.callIfCanErr(&k[l], filew, "writeParams");
        }
    }
}

const uLayer = union(enum) {
    LayerG: usize,
    LayerB: usize,
    Layer: usize,
    Relu: void,
    Pyramid: void,
    Gaussian: void,
    PGaussian: void,
    Reloid: void,
};
const Layer = union(enum) {
    LayerG: layerG,
    LayerB: layerB,
    Layer: layer,

    Relu: relu,
    Pyramid: pyramid,
    Gaussian: gaussian,
    PGaussian: pGaussian,
    Reloid: reloid,

    fn forward(this: *@This(), args: anytype) void {
        switch (this.*) {
            inline else => |*l| l.forward(args),
        }
    }
    fn backwards(this: *@This(), args: anytype) void {
        switch (this.*) {
            inline else => |*l| l.backwards(args),
        }
    }
    fn applyGradients(this: *@This(), args: anytype) void {
        utils.callIfCan(this, args, "applyGradients");
    }
};

fn layerInit(alloc: std.mem.Allocator, comptime desc: uLayer, lcommon: anytype) !Layer {
    //comptime var lsize = 0;

    const lt = switch (desc) {
        .Layer => layer,
        .LayerB => layerB,
        .LayerG => layerG,
        .Reloid => reloid,
        .Relu => relu,
        .Gaussian => gaussian,
        .PGaussian => pGaussian,
        .Pyramid => pyramid,
    };
    const lconf = switch (desc) {
        inline else => |s| s,
    };
    const ltt = switch (lt) {
        inline else => |*l| try l.init(
            alloc,
            lcommon,
            lconf,
        ),
    };

    const layerType = switch (desc) {
        .Layer => Layer{ .Layer = ltt },
        .LayerB => Layer{ .LayerB = ltt },
        .LayerG => Layer{ .LayerG = ltt },
        .Relu => Layer{ .Relu = ltt },
        .Reloid => Layer{ .Reloid = ltt },
        .Pyramid => Layer{ .Pyramid = ltt },
        .Gaussian => Layer{ .Gaussian = ltt },
        .PGaussian => Layer{ .PGaussian = ltt },
    };

    return layerType;
}

pub fn shuffleWindows(r: anytype, comptime T: type, comptime size: usize, buf: []T) void {
    const MinInt = usize;
    if (buf.len < 2) {
        return;
    }
    // `i <= j < max <= maxInt(MinInt)`
    const max: MinInt = @intCast(buf.len / size);
    var i: MinInt = 0;
    while (i < max - 1) : (i += 1) {
        const j: MinInt = @intCast(r.random().intRangeLessThan(usize, i, max));
        std.mem.swap([size]T, buf[i..][0..size], buf[j..][0..size]);
    }
}

pub fn Neuralnet(
    validationStorage: []Layer,
    storage: []Layer,
    dataset: anytype,
    config: anytype,
    allocator: std.mem.Allocator,
) ![]Layer {
    const batchSize = config.batchSize;
    var weights = std.ArrayList([]f64).init(allocator);
    var loss = try nll.init(dataset.outputSize, batchSize, allocator);

    for (storage) |*current| {
        switch (current.*) {
            inline else => |*currentLayer| {
                if (@hasField(@TypeOf(current.*), "weights")) {
                    try weights.append(currentLayer.weights);
                }
            },
        }
    }

    const t = std.time.milliTimestamp();

    var r = std.Random.DefaultPrng.init(245);

    // Do training
    for (0..config.epochs) |e| {
        shuffleWindows(&r, f64, dataset.inputSize, dataset.train_images);
        // Do training
        for (0..dataset.trainSize / batchSize) |i| {

            // Prep inputs and targets
            const inputs = dataset.train_images[i * dataset.inputSize * batchSize .. (i + 1) * dataset.inputSize * batchSize];
            const targets = dataset.train_labels[i * batchSize .. (i + 1) * batchSize];

            // Go forward and get loss

            var previousLayerOut = inputs;
            for (storage) |*current| {
                switch (current.*) {
                    inline else => |*currentLayer| {
                        currentLayer.forward(previousLayerOut);
                        previousLayerOut = currentLayer.fwd_out;
                    },
                }
            }
            //if (i % (60000 / batchSize) == 1) {
            //    std.debug.print("{any}\n", .{
            //        averageArray(loss.loss),
            //    });
            //}

            loss.getLoss(
                previousLayerOut,
                targets,
                weights.items,
                .{ .lambda = l2_lambda, .regDim = regDim },
            ) catch |err| {
                //std.debug.print("batch number: {}, time delta: {}ms\n", .{ i * batchSize, std.time.milliTimestamp() - t });
                std.debug.print("loss err: {any}\n", .{
                    utils.stats(loss.loss).avg,
                });
                return err;
            };
            var previousGradient = loss.input_grads;
            for (0..storage.len) |ni| {
                const index = storage.len - ni - 1;
                switch (storage[index]) {
                    inline else => |*currentActivation| {
                        currentActivation.backwards(previousGradient);
                        previousGradient = currentActivation.bkw_out;
                    },
                }
            }
            //use last gradient as a scalar for an optimizer?
            // Update network
            //utils.stats(previousGradient).avgabs;
            const ep = @as(f64, @floatFromInt(e + config.from));

            const adj1 = 1.0 / (@trunc(ep / 10) + 1.0);
            const adjideal1 = 1.0 / (@trunc((ep) / 10) + 1.0 / @sqrt(0.6));
            const adj2 = 1.0 / (ep / 10.0 + 1.0);
            const adjideal2 = 1.0 / (ep / 10.0 + 1.0 / @sqrt(0.6));
            //const adj3 = 1 / (@as(f64, @floatFromInt(e)) / std.math.phi + 1);
            // std.debug.print("time total: {}ms\n", .{std.time.milliTimestamp() - t});
            _ = .{ e, adj2, adj1, adjideal1, adjideal2 };
            //std.debug.print("{}", .{1 / utils.primes100[e / 10]});
            //const primeadj = 1 / utils.primes100[e / 10];

            const lr = 0.01 * adj2 * adj2;
            for (storage) |*current| {
                current.applyGradients(.{ .lambda = l2_lambda, .lr = lr, .regDim = regDim });
            }
        }

        // Do validation
        var correct: f64 = 0;
        const inputs = dataset.test_images;

        for (validationStorage, 0..) |*current, cur| {
            switch (current.*) {
                .Layer => |*currentLayer| {
                    currentLayer.copyParams(storage[cur].Layer);
                },
                .LayerB => |*currentLayer| {
                    currentLayer.copyParams(storage[cur].LayerB);
                },
                .LayerG => |*currentLayer| {
                    currentLayer.copyParams(storage[cur].LayerG);
                },
                else => {},
            }
        }
        var previousLayerOut = inputs;
        for (validationStorage) |*current| {
            switch (current.*) {
                inline else => |*currentLayer| {
                    currentLayer.forward(previousLayerOut);
                    previousLayerOut = currentLayer.fwd_out;
                },
            }
        }

        for (0..dataset.validationSize) |b| {
            var max_guess: f64 = std.math.floatMin(f64);
            var guess_index: usize = 0;
            for (previousLayerOut[b * dataset.outputSize .. (b + 1) * dataset.outputSize], 0..) |o, oi| {
                if (o > max_guess) {
                    max_guess = o;
                    guess_index = oi;
                }
            }
            if (guess_index == dataset.test_labels[b]) {
                correct += 1;
            }
        }
        correct = correct / @as(f64, @floatFromInt(dataset.validationSize));
        if (timer) {
            std.debug.print("time epoch: {}ms\n", .{std.time.milliTimestamp() - t});
        }

        std.debug.print("{}", .{correct});
        // std.debug.print(", l:{}", .{stats(loss.loss).avg});
        std.debug.print("\n", .{});
    }
    if (timer) {
        const ct = std.time.milliTimestamp();
        std.debug.print(" time schedule: {}ms\n", .{ct - t});
    }

    return storage;
}

test "Forward once" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var AA = std.heap.ArenaAllocator.init(gpa.allocator());
    const allocator = AA.allocator();
    defer _ = AA.deinit();

    const b = 2;
    var loss = try nll.NLL(2, b).init(allocator);

    // Create layer with custom weights
    var layer1 = try layer.Layer(2, 2, b).init(allocator);
    //allocator.free(layer1.weights);
    var custom_weights = [4]f64{ 0.1, 0.2, 0.3, 0.4 };
    //layer1.weights = custom_weights;
    layer1.setWeights(&custom_weights);

    // Test forward pass outputs
    const inputs = [_]f64{ 0.1, 0.2, 0.3, 0.4 };
    layer1.forward(&inputs);
    const outputs = layer1.outputs;
    const expected_outputs = [4]f64{
        0.07,
        0.1,
        0.15,
        0.22,
    };
    var i: u32 = 0;

    //std.debug.print("  batch outputs: {any}\n", .{outputs});
    while (i < 4) : (i += 1) {
        // std.debug.print("output: {any} , expected: {any}\n", .{ outputs[i], expected_outputs[i] });
        try std.testing.expectApproxEqRel(expected_outputs[i], outputs[i], 0.000000001);
    }

    // Test loss outputs
    var targets_array = [_]u8{ 0, 1 };
    const targets: []u8 = &targets_array;
    try loss.nll(outputs, targets);
    //allocator.free(outputs);
    const expected_loss = [2]f64{ 0.7082596763414484, 0.658759555548697 };
    i = 0;
    while (i < 2) : (i += 1) {
        try std.testing.expectApproxEqRel(loss.loss[i], expected_loss[i], 0.000000001);
    }

    // Test loss input_grads
    const expected_loss_input_grads = [4]f64{
        -5.074994375506203e-01,
        5.074994375506204e-01,
        4.8250714233361025e-01,
        -4.8250714233361025e-01,
    };
    i = 0;
    while (i < 4) : (i += 1) {
        try std.testing.expectApproxEqRel(loss.input_grads[i], expected_loss_input_grads[i], 0.000000001);
    }

    // Do layer backwards
    layer1.backwards(loss.input_grads);

    // Test layer weight grads
    const expected_layer_weight_grads = [4]f64{
        4.700109947251052e-02,
        -4.7001099472510514e-02,
        4.575148471166002e-02,
        -4.5751484711660004e-02,
    };
    i = 0;
    //std.debug.print("\n output: {any} ,\n expected: {any}\n", .{
    //    grads.weight_grads,
    //    expected_layer_weight_grads,
    //});
    while (i < 4) : (i += 1) {
        try std.testing.expectApproxEqRel(
            expected_layer_weight_grads[i],
            layer1.weight_grads[i],
            0.000_000_001,
        );
    }

    // Test layer input grads
    const expected_layer_input_grads = [4]f64{
        5.074994375506206e-02,
        5.074994375506209e-02,
        -4.8250714233361025e-02,
        -4.8250714233361025e-02,
    };
    i = 0;

    //std.debug.print("output: {any} , expected: {any}\n", .{
    //    grads.input_grads,
    //    expected_layer_input_grads,
    //});

    while (i < 4) : (i += 1) {
        //std.debug.print("output: {any} , expected: {any}\n", .{
        //    grads.input_grads[i],
        //    expected_layer_input_grads[i],
        //});
        try std.testing.expectApproxEqRel(layer1.input_grads[i], expected_layer_input_grads[i], 0.000000001);
    }
}

//test "Train Memory Leak" {
//    var allocator = std.testing.allocator;
//
//    // Get MNIST data
//    const mnist_data = try mnist.readMnist(allocator);
//
//    // Prep loss function
//    const loss_function = nll.NLL(OUTPUT_SIZE);
//
//    // Prep NN
//    var layer1 = try layer.Layer(INPUT_SIZE, 100).init(allocator);
//    var relu1 = relu.Relu.new();
//    var layer2 = try layer.Layer(100, OUTPUT_SIZE).init(allocator);
//
//    // Prep inputs and targets
//    const inputs = mnist_data.train_images[0..INPUT_SIZE];
//    const targets = mnist_data.train_labels[0..1];
//
//    // Go forward and get loss
//    const outputs1 = try layer1.forward(inputs, allocator);
//    const outputs2 = try relu1.forward(outputs1, allocator);
//    const outputs3 = try layer2.forward(outputs2, allocator);
//    const loss = try loss_function.nll(outputs3, targets, allocator);
//
//    // Update network
//    const grads1 = try layer2.backwards(loss.input_grads, allocator);
//    const grads2 = try relu1.backwards(grads1.input_grads, allocator);
//    const grads3 = try layer1.backwards(grads2, allocator);
//    layer1.applyGradients(grads3.weight_grads);
//    layer2.applyGradients(grads1.weight_grads);
//
//    // Free memory
//    allocator.free(outputs1);
//    allocator.free(outputs2);
//    allocator.free(outputs3);
//    grads1.deinit(allocator);
//    allocator.free(grads2);
//    grads3.deinit(allocator);
//    loss.deinit(allocator);
//
//    layer1.deinit(allocator);
//    layer2.deinit(allocator);
//    mnist_data.deinit(allocator);
//}
