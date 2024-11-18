const std = @import("std");
const lt = @import("layerTypes.zig");

const GiveLoss = true;

loss: []f64, // = [1]f64{0} ** (batchSize);
input_grads: []f64, // = [1]f64{0} ** (batchSize * inputSize);
batchSize: usize,
inputSize: usize,

const Self = @This();

pub fn init(
    inputSize: usize,
    batchSize: usize,
    alloc: std.mem.Allocator,
) !Self {
    return Self{
        .loss = try alloc.alloc(f64, batchSize),
        .input_grads = try alloc.alloc(f64, batchSize * inputSize),
        .batchSize = batchSize,
        .inputSize = inputSize,
    };
}
const log = false;
pub fn getLoss(self: *Self, inputs: []f64, targets: []u8, network: []lt.Layer, config: anytype) !void {
    std.debug.assert(targets.len == self.batchSize);
    std.debug.assert(inputs.len == self.batchSize * self.inputSize);

    //_ = (lambda / 2.0) * l2_sum;

    var a_sum: f64 = 0.0;
    var lp_sum: f64 = 0.0;

    //std.debug.print("\n {}", .{network.len});
    for (network, 0..) |layer, li| {
        //std.debug.print(" {}", .{li});
        switch (layer) {
            inline else => |lfilt| {
                if (!@hasField(@TypeOf(lfilt), "weights")) continue;

                const weightCount = @as(f64, @floatFromInt(lfilt.inputSize * lfilt.outputSize));
                const osize = @as(f64, @floatFromInt(lfilt.outputSize));
                if (@hasField(@TypeOf(lfilt), "fwd_out")) {
                    for (lfilt.fwd_out, 0..) |a, oi| {
                        const nw = @as(f64, @floatFromInt(network.len));
                        const lin = nw - @as(f64, @floatFromInt(li));
                        const oin = @as(f64, @floatFromInt(oi));
                        _ = .{ oin, lin };
                        //std.debug.print(" {}", .{li});
                        const poswf = lin / nw; // (oin + 1);
                        a_sum += @abs((a * poswf) / osize);
                    }
                }
                const weights = if (@hasField(@TypeOf(lfilt.weights), "data"))
                    lfilt.weights.data
                else
                    lfilt.weights;

                for (weights) |weight| {
                    lp_sum += std.math.pow(f64, @abs(weight / weightCount), config.regDim);
                }
            },
        }
        _ = .{a_sum};
    }
    const aterm = config.lambda * a_sum; // Equivalent to the Lp penalty term
    const lp_term = (config.lambda / config.regDim) * lp_sum + aterm;
    //const l2_term = (lambda / 2.0) * l2_sum;

    //todo make assert right.
    if (inputs.len != 10 * 100) std.debug.print("should be equal {any} in, expect {any}", .{ inputs.len, 10 * 100 });
    for (0..self.batchSize) |b| {
        var sum: f64 = 0;
        var maxInput: f64 = -std.math.inf(f64);
        // Find the maximum value in the inputs for the current batch
        for (0..self.inputSize) |i| {
            maxInput = @max(maxInput, inputs[b * self.inputSize + i]);
        }
        for (0..self.inputSize) |i| {
            sum += std.math.exp(inputs[b * self.inputSize + i] - maxInput);
            if (sum == std.math.inf(f64)) {
                std.debug.print("output with inf:\n {any},\n", .{
                    inputs[b * self.inputSize + i],
                });
                return error.divisionbyinf;
            }
        }
        const s = sum;
        //std.math.sign(sum + 0.0000001) *
        sum = (std.math.sign(sum) + 0.000000001) * @max(0.0000001, @abs(sum));
        if (sum >= std.math.inf(f64)) return error.divisionbyinf;
        if (std.math.sign(sum) == 0) {
            std.debug.print("sum:{}\n", .{s});
            return error.sumisnan;
        }
        if (sum == 0) {
            return error.divisionbyzero;
        }

        if (GiveLoss) {
            //std.debug.print("target:{}\n", .{targets[b]});
            self.loss[b] = -1 * @log(std.math.exp(inputs[b * self.inputSize + targets[b]]) / sum);
        }
        for (0..self.inputSize) |i| {
            self.input_grads[b * self.inputSize + i] = std.math.exp(inputs[b * self.inputSize + i] - maxInput) / sum + lp_term;
            if (i == targets[b]) {
                self.input_grads[b * self.inputSize + i] -= 1;
            }
        }
    }
}
