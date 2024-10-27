const std = @import("std");

const utils = @import("utils.zig");

last_inputs: []const f64 = undefined,
fwd_out: []f64,
bkw_out: []f64,
loss: []f64,
batchSize: usize,
size: usize,

const Self = @This();

pub fn init(
    alloc: std.mem.Allocator,
    lcommon: struct {
        batchSize: usize,
        inputSize: usize,
    },
    _: anytype,
) !Self {
    const size = lcommon.inputSize;
    const batchSize = lcommon.batchSize;

    return Self{
        .last_inputs = try alloc.alloc(f64, size * batchSize),
        .fwd_out = try alloc.alloc(f64, size * batchSize / 2),
        .bkw_out = try alloc.alloc(f64, size * batchSize),
        .loss = try alloc.alloc(f64, size * batchSize / 2),
        .batchSize = batchSize,
        .size = size,
    };
}
pub fn deinitBackwards(self: *Self, alloc: std.mem.Allocator) void {
    alloc.free(self.bkw_out);
}

var prng = std.Random.DefaultPrng.init(123);
const limit = std.math.floatMax(f64);

pub fn forward(self: *Self, inputs: []f64) void {
    std.debug.assert(inputs.len == self.size * self.batchSize);

    for (0..self.batchSize) |b| {
        for (0..self.size / 2) |i| {
            const idx = b * self.size + i;
            const idxo = b * self.size / 2 + i;
            //const idx2 = b * self.size + self.size / 2 + i;
            self.fwd_out[idxo] = inputs[idx];
        }
    }
    self.last_inputs = inputs;
}

pub fn backwards(self: *Self, grads: []f64) void {
    std.debug.assert(grads.len == self.size * self.batchSize / 2);
    self.makeGrads();
    self.loss = utils.normalize(self.loss, 1, 0, 1);
    for (0..self.batchSize) |b| {
        for (0..self.size / 2) |i| {
            const idx = b * self.size + i;
            const idxo = b * self.size / 2 + i;
            const idx2 = b * self.size + self.size / 2 + i;
            //TODO
            const scale = 1.0 / @as(f64, @floatFromInt(self.size * self.batchSize / 2));
            self.bkw_out[idx] = grads[idxo] + grads[idxo] * self.loss[idxo] * scale;
            self.bkw_out[idx2] = grads[idxo] * self.loss[idxo] * scale;
        }
    }
}
pub fn makeGrads(self: *Self) void {
    //const regDim = config.regDim;
    //const lambda = config.lambda;
    //_ = .{ regDim, lambda };

    std.debug.assert(self.last_inputs.len == self.batchSize * self.size);

    //TODO
    //var lp_sum: f64 = 0.0;
    //for (network) |layer| {
    //    if (@hasField(@TypeOf(layer), "weights")) {
    //        for (layer.weights) |weight| {
    //            lp_sum += std.math.pow(f64, @abs(weight), regDim);
    //        }
    //    }
    //}
    //const lp_term = (lambda / regDim) * lp_sum;

    for (0..self.batchSize) |b| {
        var batch_loss: f64 = 0.0;
        var maxInput: f64 = -std.math.inf(f64);

        // Find the maximum input value for numerical stability in softmax
        for (0..self.size / 2) |i| {
            maxInput = @max(maxInput, self.last_inputs[b * self.size + i]);
        }

        var sum: f64 = 0.0;
        for (0..self.size / 2) |i| {
            const adjustedInput = self.last_inputs[b * self.size + i] - maxInput;
            sum += std.math.exp(adjustedInput);
        }

        const softmax_denom = @max(sum, 1e-10);
        //if (softmax_denom == std.math.inf(f64)) return error.divisionbyinf;

        for (0..self.size / 2) |i| {
            const idx = b * self.size + i;
            const idx2 = b * self.size + self.size / 2 + i;
            const prob = std.math.exp(self.last_inputs[idx] - maxInput) / softmax_denom;
            const target_prob = @max(self.last_inputs[idx2], 1e-10); // Stabilize log

            // Accumulate cross-entropy loss
            batch_loss += -target_prob * @log(prob);

            // Calculate gradient for input
            self.loss[idx] = prob - target_prob;
        }

        // Store the loss for the current batch, including regularization term
        //self.loss[b] = batch_loss; // + lp_term;
    }
}
