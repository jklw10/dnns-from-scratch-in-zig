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
const dropout = @import("dropout.zig");

const lt = @import("layerTypes.zig");

const utils = @import("utils.zig");
const std = @import("std");

const timer = false;

const scheduleItem = struct {
    epochs: usize,
    hLSize: usize,
};
const schedule = [_]scheduleItem{
    //.{ .epochs = 100, .hLSize = 25 },
    //.{ .epochs = 1, .hLSize = 10 },
    //.{ .epochs = 2, .hLSize = 20 },
    //.{ .epochs = 4, .hLSize = 40 },
    //.{ .epochs = 8, .hLSize = 80 },
    .{ .epochs = 100, .hLSize = 25 },
    //.{ .epochs = 25, .hLSize = 25 },
    //.{ .epochs = 32, .hLSize = 32 },
    //.{ .epochs = 64, .hLSize = 64 },
    //.{ .epochs = 100, .hLSize = 100 },
};

const resetEpOnRescale = true;
const continueFrom = 0;
const l2_lambda = 0.0075;
const m = std.math;
const regDim: f64 = m.phi;

const graphfuncs = false;
const multiIter = true;

const fileSignature = "G25RRRR_G10R.f64";

const reinit = false;

pub fn main() !void {

    //var asd = try @import("clKernel.zig").init();
    //try asd.run();
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};

    const allocator = gpa.allocator();

    const dataset = try dataSet.mnist.dtype.readData(allocator);
    defer dataset.deinit(allocator);

    if (graphfuncs) {
        utils.graphFunc(gaussian);
    }

    std.debug.print("Training... \n", .{});

    const t = std.time.milliTimestamp();
    //try runSchedule(1, dataset, allocator);

    inline for (0..schedule.len) |itera| {
        //std.debug.print("sched {} \n", .{itera});
        var arena = std.heap.ArenaAllocator.init(allocator);
        try runSchedule(itera, dataset, arena.allocator());
        arena.deinit();
    }

    const ct = std.time.milliTimestamp();

    std.debug.print("time total: {}ms\n", .{ct - t});
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
    if (!resetEpOnRescale and !first) {
        for (schedule[0 .. itera - 1]) |elem| {
            sum += elem.epochs;
        }
    }
    const pastEp = sum;

    //std.debug.print("sched {} \n", .{pastEp});

    const default = lt.uLayer.Relu;

    const fileL = [_]lt.uLayer{
        .{ .LayerG = ps },     default,
        .{ .LayerG = ps * 2 }, .Reloid,
        .SelfPredict,          .{ .LayerG = ps * 2 },
        .Reloid,               .SelfPredict,
        .{ .LayerG = ps * 2 }, .Reloid,
        .SelfPredict,          .{ .LayerG = 10 * 2 },
        default,               .SelfPredict,
    };
    const layers = comptime [_]lt.uLayer{
        .{ .LayerG = cs },     default, //.SelfPredict,
        .{ .LayerG = cs * 2 }, .Reloid,
        .SelfPredict,          .{ .LayerG = cs * 2 },
        .Reloid,               .SelfPredict,
        .{ .LayerG = cs * 2 }, .Reloid,
        .SelfPredict,          .{ .LayerG = 10 * 2 },
        default,               .SelfPredict,
    };

    const nntype = LayerStorage(&layers, @TypeOf(dataset));

    var neuralNet = try nntype.init(
        .{ .batchSize = batchSize },
        allocator,
    );

    if (readfile) {
        const file = try std.fs.cwd().createFile(
            "data/Params_" ++ fileSignature,
            .{
                .read = true,
                .truncate = false,
            },
        );
        var reader = std.io.bufferedReader(file.reader());
        defer file.close();
        if (resize) {
            var other = try LayerStorage(&fileL, @TypeOf(dataset)).init(
                .{ .deinitbkw = false, .batchSize = batchSize },
                allocator,
            );
            try other.fromFile(&reader);
            try neuralNet.rescale(other);
        } else {
            try neuralNet.fromFile(&reader);
        }
    }

    try neuralNet.Run(
        dataset,
        .{ .epochs = epochs, .batchSize = batchSize, .from = pastEp },
    );
    if (writeFile) {
        try neuralNet.toFile();
    }
}
fn LayerStorage(definition: []const lt.uLayer, datatype: anytype) type {
    return struct {
        validationStorage: []lt.Layer,
        storage: []lt.Layer,
        loss: nll,

        comptime definition: []const lt.uLayer = definition,
        comptime datatype: type = datatype,
        const Self = @This();
        fn init(config: anytype, allocator: std.mem.Allocator) !Self {
            comptime var previousLayerSize = datatype.inputSize;
            var storage: []lt.Layer = try allocator.alloc(lt.Layer, definition.len);
            var validationStorage: []lt.Layer = try allocator.alloc(lt.Layer, definition.len);
            inline for (definition, 0..) |lay, i| {
                storage[i] = try lay.layerInit(
                    allocator,
                    .{
                        .batchSize = config.batchSize,
                        .inputSize = previousLayerSize,
                    },
                );
                validationStorage[i] = try lay.layerInit(
                    allocator,
                    .{
                        .batchSize = datatype.validationSize,
                        .inputSize = previousLayerSize,
                    },
                );
                switch (validationStorage[i]) {
                    inline else => |*l| l.deinitBackwards(allocator),
                }
                switch (lay) {
                    .Layer, .LayerB, .LayerG => |size| previousLayerSize = size,
                    .PGaussian => previousLayerSize = previousLayerSize - 3,
                    .SelfPredict => previousLayerSize = previousLayerSize / 2,
                    else => {},
                }
            }
            return .{
                .storage = storage,
                .validationStorage = validationStorage,
                .loss = try nll.init(datatype.outputSize, config.batchSize, allocator),
            };
        }

        fn fromFile(self: *Self, reader: anytype) !void {
            for (self.storage) |*lay| {
                try utils.callIfCanErr(lay, reader, "readParams");
            }
        }

        fn toFile(self: Self) !void {
            const filew = try std.fs.cwd().createFile(
                "data/Params_" ++ fileSignature,
                .{
                    .read = true,
                    .truncate = true,
                },
            );
            defer filew.close();
            for (self.storage) |l| {
                //l.writeParams(filew);
                try utils.callIfCanErr(&l, filew, "writeParams");
            }
        }
        fn reinit(self: *Self) !void {
            for (self.storage) |lay| {
                try utils.callIfCanErr(&lay, 1.0, "reinit");
            }
        }
        fn rescale(self: Self, other: anytype) !void {
            for (other.storage, 0..) |_, i| {
                utils.callIfTypeMatch(&self.storage[i], other, "rescale");
            }
        }
        fn Run(
            self: *Self,
            dataset: anytype,
            config: anytype,
        ) !void {
            const batchSize = config.batchSize;
            var storage = self.storage;
            const validationStorage = self.validationStorage;

            const t = std.time.milliTimestamp();

            var r = std.Random.DefaultPrng.init(245);

            // Do training
            for (0..config.epochs) |e| {
                utils.shufflePairedWindows(
                    &r,
                    f64,
                    datatype.inputSize,
                    dataset.train_images,
                    u8,
                    1,
                    dataset.train_labels,
                );
                // Do training
                for (0..dataset.trainSize / batchSize) |i| {

                    // Prep inputs and targets
                    const inputs = dataset.train_images[i * datatype.inputSize * batchSize .. (i + 1) * datatype.inputSize * batchSize];
                    const targets = dataset.train_labels[i * batchSize .. (i + 1) * batchSize];

                    // Go forward and get loss

                    var previousLayerOut = inputs;
                    for (storage) |*current| {
                        switch (current.*) {
                            inline else => |*currentLayer| {
                                //if (i == 0) std.debug.print("size {any}", .{previousLayerOut.len});

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

                    self.loss.getLoss(
                        previousLayerOut,
                        targets,
                        self.storage,
                        .{ .lambda = l2_lambda, .regDim = regDim },
                    ) catch |err| {
                        //std.debug.print("batch number: {}, time delta: {}ms\n", .{ i * batchSize, std.time.milliTimestamp() - t });
                        std.debug.print("loss err: {any}\n", .{
                            utils.stats(self.loss.loss).avg,
                        });
                        return err;
                    };
                    var previousGradient = self.loss.input_grads;
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
                    if (i == 0) {
                        //std.debug.print("e: {} ", .{ep});
                    }

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
                            //std.debug.print("size {any}", .{previousLayerOut.len});
                            currentLayer.forward(previousLayerOut);
                            previousLayerOut = currentLayer.fwd_out;
                        },
                    }
                }

                for (0..datatype.validationSize) |b| {
                    var max_guess: f64 = std.math.floatMin(f64);
                    var guess_index: usize = 0;
                    for (previousLayerOut[b * datatype.outputSize .. (b + 1) * datatype.outputSize], 0..) |o, oi| {
                        if (o > max_guess) {
                            max_guess = o;
                            guess_index = oi;
                        }
                    }
                    if (guess_index == dataset.test_labels[b]) {
                        correct += 1;
                    }
                }
                correct = correct / @as(f64, @floatFromInt(datatype.validationSize));
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
        }
    };
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
