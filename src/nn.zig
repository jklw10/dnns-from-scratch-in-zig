const lt = @import("layerTypes.zig");
const utils = @import("utils.zig");
const std = @import("std");

pub fn NeuralNetwork(datatype: anytype, losst: anytype) type {
    return struct {
        storage: []lt.Layer,
        loss: losst,
        batchSize: usize,
        comptime datatype: type = datatype,
        const Self = @This();

        var r = std.Random.DefaultPrng.init(245);
        pub fn init(comptime definition: []const lt.uLayer, config: anytype, allocator: std.mem.Allocator) !Self {
            comptime var previousLayerSize = datatype.inputSize;
            var storage: []lt.Layer = try allocator.alloc(lt.Layer, definition.len);
            inline for (definition, 0..) |lay, i| {
                storage[i] = try lay.layerInit(
                    allocator,
                    .{
                        .batchSize = config.batchSize,
                        .inputSize = previousLayerSize,
                    },
                );
                if (config.deinitBackwards) {
                    try utils.callIfCanErr(&storage[i], allocator, "deinitBackwards");
                }
                switch (lay) {
                    .Layer, .LayerB, .LayerG => |size| previousLayerSize = size,
                    .PGaussian => previousLayerSize = previousLayerSize - 3,
                    else => {},
                }
            }
            return .{
                .storage = storage,
                .loss = try losst.init(datatype.outputSize, config.batchSize, config, allocator),
                .batchSize = config.batchSize,
            };
        }
        pub fn copyParams(self: *Self, other: Self) void {
            for (self.storage, 0..) |*current, cur| {
                switch (current.*) {
                    .Layer => |*currentLayer| {
                        currentLayer.copyParams(other.storage[cur].Layer);
                    },
                    .LayerB => |*currentLayer| {
                        currentLayer.copyParams(other.storage[cur].LayerB);
                    },
                    .LayerG => |*currentLayer| {
                        currentLayer.copyParams(other.storage[cur].LayerG);
                    },
                    else => {},
                }
            }
        }
        const fileSignature = "";
        pub fn fromFile(self: *Self) !void {
            const file = try std.fs.cwd().createFile(
                "data/Params_" ++ fileSignature,
                .{
                    .read = true,
                    .truncate = false,
                },
            );
            var reader = std.io.bufferedReader(file.reader());
            defer file.close();
            for (self.storage) |*lay| {
                try utils.callIfCanErr(lay, &reader, "readParams");
            }
        }

        pub fn toFile(self: Self) !void {
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
        pub fn reinit(self: *Self) !void {
            for (self.storage) |lay| {
                try utils.callIfCanErr(&lay, 1.0, "reinit");
            }
        }
        pub fn insert(self: *Self, other: anytype) !void {
            for (other.storage, 0..) |_, i| {
                utils.callIfTypeMatch(&self.storage[i], other, "insert");
            }
        }
        pub fn batchForward(
            self: *Self,
            batch: []f64,
        ) []f64 {
            var previousLayerOut = batch;
            for (self.storage) |*current| {
                switch (current.*) {
                    inline else => |*currentLayer| {
                        currentLayer.forward(previousLayerOut);
                        previousLayerOut = currentLayer.fwd_out;
                    },
                }
            }
            return previousLayerOut;
        }
        pub fn batchBackwards(
            self: *Self,
        ) void {
            var previousGradient = self.loss.input_grads;
            for (0..self.storage.len) |ni| {
                const index = self.storage.len - ni - 1;
                switch (self.storage[index]) {
                    inline else => |*currentActivation| {
                        currentActivation.backwards(previousGradient);
                        previousGradient = currentActivation.bkw_out;
                    },
                }
            }
        }
        pub fn validate(
            self: *Self,
            dataset: anytype,
        ) f64 {
            var previousLayerOut = self.batchForward(dataset.test_images);
            var correct: f64 = 0;
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
            return correct / @as(f64, @floatFromInt(datatype.validationSize));
        }
        pub fn epoch(self: *Self, dataset: anytype, config: anytype) !void {
            // utils.shuffleWindows(&r, f64, datatype.inputSize, dataset.train_images);
            utils.shufflePairedWindows(
                &r,
                f64,
                datatype.inputSize,
                dataset.train_images,
                u8,
                1,
                dataset.train_labels,
            );
            for (0..dataset.trainSize / self.batchSize) |i| {
                const inputs = dataset.train_images[i * datatype.inputSize * self.batchSize ..][0 .. datatype.inputSize * self.batchSize];
                const targets = dataset.train_labels[i * self.batchSize ..][0..self.batchSize];

                const fwdout = self.batchForward(inputs);

                self.loss.getLoss(
                    fwdout,
                    targets,
                    self.storage,
                ) catch |err| {
                    //std.debug.print("batch number: {}, time delta: {}ms\n", .{ i * batchSize, std.time.milliTimestamp() - t });
                    std.debug.print("loss err: {any}\n", .{
                        utils.stats(self.loss.loss).avg,
                    });
                    return err;
                };
                self.batchBackwards();
                const ep = @as(f64, @floatFromInt(config.epoch));
                const adj2 = 1.0 / (ep / 10.0 + 1.0);

                const lr = 0.01 * adj2 * adj2;
                for (self.storage) |*current| {
                    current.applyGradients(.{ .lambda = config.lambda, .lr = lr, .regDim = config.regDim });
                }
            }
        }
    };
}
