const layer = @import("layer.zig");
const layerB = @import("layerBias.zig");
const layerG = @import("layerGrok.zig");
const nll = @import("nll.zig");
const dataSet = @import("dataSet.zig");
const relu = @import("relu.zig");
const reloid = @import("reloid.zig");
const pyramid = @import("pyramid.zig");
const gaussian = @import("gaussian.zig");

const std = @import("std");
const timer = false;

const readfile = false;
const writeFile = true;

const typesignature = "G25RRRR_G10R.f64";

const graphfuncs = false;
const reinit = false;

const epoch = 100;
//todo perlayer configs

pub fn main() !void {
    const batchSize = 100;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};

    const allocator = gpa.allocator();

    // Get MNIST data

    const dataset = if (true) blk: {
        const mnist_train_images_path: []const u8 = "data/mnist/train-images.idx3-ubyte";
        const mnist_train_labels_path: []const u8 = "data/mnist/train-labels.idx1-ubyte";
        const mnist_test_images_path: []const u8 = "data/mnist/t10k-images.idx3-ubyte";
        const mnist_test_labels_path: []const u8 = "data/mnist/t10k-labels.idx1-ubyte";

        const inputSize = 784;
        const outputSize = 10;
        const dataSize = 60000;
        const validationSize = 10000;
        break :blk try dataSet.readData(
            mnist_train_images_path,
            mnist_train_labels_path,
            mnist_test_images_path,
            mnist_test_labels_path,
            dataSize,
            inputSize,
            outputSize,
            validationSize,
            allocator,
        );
    } else blk: {
        const mnist_train_images_path: []const u8 = "data/mnist/train-images.idx3-ubyte";
        const mnist_train_labels_path: []const u8 = "data/mnist/train-labels.idx1-ubyte";
        const mnist_test_images_path: []const u8 = "data/mnist/t10k-images.idx3-ubyte";
        const mnist_test_labels_path: []const u8 = "data/mnist/t10k-labels.idx1-ubyte";

        const inputSize = 784;
        const outputSize = 10;
        const dataSize = 60000;
        const validationSize = 10000;
        break :blk try dataSet.readData(
            mnist_train_images_path,
            mnist_train_labels_path,
            mnist_test_images_path,
            mnist_test_labels_path,
            dataSize,
            inputSize,
            outputSize,
            validationSize,
            allocator,
        );
    };

    //const dataset = try dataSet.readMnist(allocator);
    defer dataset.deinit(allocator);

    if (graphfuncs) {
        var inputs: [200]f64 = undefined;
        var pyr = try gaussian.init(allocator, 1, 200);
        for (inputs, 0..) |_, i| {
            inputs[i] = (-100 + @as(f64, @floatFromInt(i))) / 20;
        }

        pyr.forward(&inputs);
        pyr.backwards(&inputs);
        for (inputs, 0..) |_, i| {
            std.debug.print("{d:.4},", .{inputs[i]});
            std.debug.print("{d:.4},", .{gaussian.leaky_gaussian(inputs[i])});
            std.debug.print("{d:.4}\n", .{gaussian.leaky_gaussian_derivative(inputs[i])});
        }
    }

    const file = try std.fs.cwd().createFile(
        "data/Params_" ++ typesignature,
        .{
            .read = true,
            .truncate = false,
        },
    );

    const default = uActivation.relu;
    const layers = [_]layerDescriptor{ .{
        .layer = .{ .LayerG = 25 },
        .activation = default,
    }, .{
        .layer = .{ .LayerG = 25 },
        .activation = .reloid,
    }, .{
        .layer = .{ .LayerG = 25 },
        .activation = .reloid,
    }, .{
        .layer = .{ .LayerG = 25 },
        .activation = .reloid,
    }, .{
        .layer = .{ .LayerG = 10 },
        .activation = default,
    } };
    var previousLayerSize = dataset.inputSize;
    var storage: [layers.len]layerStorage = undefined;
    var validationStorage: [layers.len]layerStorage = undefined;
    //var reader = std.io.limitedReader(file.reader(), (try file.stat()).size);
    std.debug.assert(dataset.outputSize == switch (layers[storage.len - 1].layer) {
        .Layer, .LayerB, .LayerG => |l| l,
    });
    var reader = std.io.bufferedReader(file.reader());
    // Prep NN
    inline for (layers, 0..) |lay, i| {
        const size = switch (lay.layer) {
            .Layer, .LayerB, .LayerG => |size| size,
        };
        storage[i] = try layerFromDescriptor(
            allocator,
            lay,
            batchSize,
            previousLayerSize,
        );
        validationStorage[i] = try layerFromDescriptor(
            allocator,
            lay,
            dataset.validationSize,
            previousLayerSize,
        );
        switch (validationStorage[i].layer) {
            inline else => |*l| l.deinitBackwards(allocator),
        }
        if (readfile) {
            switch (storage[i].layer) {
                .LayerG => |*l| {
                    try l.readParams(&reader);
                    if (reinit) {
                        l.reinit(1.000);
                    }
                },
                inline else => |*l| {
                    try l.readParams(&reader);
                },
            }
        }
        previousLayerSize = size;
    }

    file.close();

    const k = try Neuralnet(
        &validationStorage,
        &storage,
        batchSize,
        epoch,
        dataset,
        allocator,
    );

    if (writeFile) {
        const filew = try std.fs.cwd().createFile(
            "data/Params_" ++ typesignature,
            .{
                .read = true,
                .truncate = true,
            },
        );
        defer filew.close();
        for (0..k.len) |l| {
            switch (k[l].layer) {
                inline else => |*la| {
                    try la.writeParams(filew);
                },
            }
        }
    }
}

const uActivation = enum {
    none,
    relu,
    pyramid,
    gaussian,
    reloid,
};

const Activation = union(uActivation) {
    none: void,
    relu: relu,
    pyramid: pyramid,
    gaussian: gaussian,
    reloid: reloid,
};

const uLayer = union(enum) {
    LayerG: usize,
    LayerB: usize,
    Layer: usize,
};
const Layer = union(enum) {
    LayerG: layerG,
    LayerB: layerB,
    Layer: layer,
};

const layerDescriptor = struct {
    layer: uLayer,
    activation: uActivation,
};

const layerStorage = struct {
    layer: Layer,
    activation: Activation,
};

fn layerFromDescriptor(alloc: std.mem.Allocator, comptime desc: layerDescriptor, batchSize: usize, inputSize: usize) !layerStorage {
    //comptime var lsize = 0;

    const lt = switch (desc.layer) {
        .Layer => layer,
        .LayerB => layerB,
        .LayerG => layerG,
    };
    const lsize = switch (desc.layer) {
        inline else => |s| s,
    };
    const ltt = switch (lt) {
        inline else => |*l| try l.init(
            alloc,
            batchSize,
            inputSize,
            lsize,
        ),
    };
    const layerType = switch (desc.layer) {
        .Layer,
        => Layer{ .Layer = ltt },
        .LayerB => Layer{ .LayerB = ltt },
        .LayerG => Layer{ .LayerG = ltt },
    };
    const activation = switch (desc.activation) {
        .relu => Activation{
            .relu = try relu.init(
                alloc,
                batchSize,
                lsize,
            ),
        },
        .reloid => Activation{
            .reloid = try reloid.init(
                alloc,
                batchSize,
                lsize,
            ),
        },
        .pyramid => Activation{
            .pyramid = try pyramid.init(
                alloc,
                batchSize,
                lsize,
            ),
        },
        .gaussian => Activation{
            .gaussian = try gaussian.init(
                alloc,
                batchSize,
                lsize,
            ),
        },
        .none => .none,
    };

    return .{
        .layer = layerType,
        .activation = activation,
    };
}

pub fn Neuralnet(
    //comptime layers: []const layerDescriptor,
    validationStorage: []layerStorage,
    storage: []layerStorage,
    comptime batchSize: u32,
    comptime epochs: u32,
    dataset: dataSet.Data,
    allocator: std.mem.Allocator,
) ![]layerStorage {

    //const testImageCount = 10000;

    var loss = try nll.init(dataset.outputSize, batchSize, allocator);

    const t = std.time.milliTimestamp();
    std.debug.print("Training... \n", .{});
    // Do training
    for (0..epochs) |_| {
        //if (e % 50 == 0) {
        //    std.debug.print("Reinit \n", .{});
        //    for (storage, 0..) |_, i| {
        //        switch (storage[i].layer) {
        //            .LayerG => |*l| {
        //                l.reinit();
        //            },
        //            else => {},
        //        }
        //    }
        //}
        // Do training

        for (0..dataset.trainSize / batchSize) |i| {

            // Prep inputs and targets
            const inputs = dataset.train_images[i * dataset.inputSize * batchSize .. (i + 1) * dataset.inputSize * batchSize];
            const targets = dataset.train_labels[i * batchSize .. (i + 1) * batchSize];

            // Go forward and get loss

            var previousLayerOut = inputs;
            for (storage) |*current| {
                switch (current.layer) {
                    inline else => |*currentLayer| {
                        currentLayer.forward(previousLayerOut);
                        previousLayerOut = currentLayer.outputs;
                    },
                }
                switch (current.activation) {
                    .none => {},
                    inline else => |*currentActivation| {
                        currentActivation.forward(previousLayerOut);
                        previousLayerOut = currentActivation.fwd_out;
                    },
                }
            }
            //if (i % (60000 / batchSize) == 1) {
            //    std.debug.print("{any}\n", .{
            //        averageArray(loss.loss),
            //    });
            //}

            loss.nll(previousLayerOut, targets) catch |err| {
                //std.debug.print("batch number: {}, time delta: {}ms\n", .{ i * batchSize, std.time.milliTimestamp() - t });
                std.debug.print("loss err: {any}\n", .{
                    averageArray(loss.loss),
                });
                return err;
            };
            var previousGradient = loss.input_grads;
            for (0..storage.len) |ni| {
                const index = storage.len - ni - 1;
                switch (storage[index].activation) {
                    .none => {},
                    inline else => |*currentActivation| {
                        currentActivation.backwards(previousGradient);
                        previousGradient = currentActivation.bkw_out;
                    },
                }
                switch (storage[index].layer) {
                    inline else => |*currentLayer| {
                        currentLayer.backwards(previousGradient);
                        previousGradient = currentLayer.input_grads;
                    },
                }
            }
            // Update network

            for (storage) |*current| {
                switch (current.layer) {
                    inline else => |*currentLayer| {
                        currentLayer.applyGradients();
                    },
                }
            }
        }

        // Do validation
        var correct: f64 = 0;
        const inputs = dataset.test_images;

        for (validationStorage, 0..) |*current, cur| {
            switch (current.layer) {
                .Layer => |*currentLayer| {
                    currentLayer.setWeights(storage[cur].layer.Layer.weights);
                },
                .LayerB => |*currentLayer| {
                    currentLayer.setWeights(storage[cur].layer.LayerB.weights);
                    currentLayer.setBiases(storage[cur].layer.LayerB.biases);
                },
                .LayerG => |*currentLayer| {
                    currentLayer.setWeights(storage[cur].layer.LayerG.weights);
                    currentLayer.setBiases(storage[cur].layer.LayerG.biases);
                },
            }
        }
        var previousLayerOut = inputs;
        for (validationStorage) |*current| {
            switch (current.layer) {
                inline else => |*currentLayer| {
                    currentLayer.forward(previousLayerOut);
                    previousLayerOut = currentLayer.outputs;
                },
            }
            switch (current.activation) {
                .none => {},
                inline else => |*currentActivation| {
                    currentActivation.forward(previousLayerOut);
                    previousLayerOut = currentActivation.fwd_out;
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
            std.debug.print("time total: {}ms\n", .{std.time.milliTimestamp() - t});
        }

        std.debug.print("{}\n", .{correct});
    }
    const ct = std.time.milliTimestamp();
    std.debug.print(" time total: {}ms\n", .{ct - t});
    return storage;
}
fn averageArray(arr: []f64) f64 {
    var sum: f64 = 0;
    for (arr) |elem| {
        sum += elem;
    }
    return sum / @as(f64, @floatFromInt(arr.len));
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
