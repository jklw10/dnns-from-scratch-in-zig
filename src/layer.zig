const std = @import("std");

weights: []f64,
last_inputs: []const f64,
fwd_out: []f64,
weight_grads: []f64, // = [1]f64{0} ** (inputSize * outputSize);
bkw_out: []f64, //= [1]f64{0} ** (batchSize * inputSize);
batchSize: usize,
inputSize: usize,
outputSize: usize,

const Self = @This();
//var outputs: [batchSize * outputSize]f64 = [1]f64{0} ** (batchSize * outputSize);
pub fn copyParams(self: *Self, other: Self) void {
    self.weights = other.weights;
}

pub fn readParams(self: *Self, params: anytype) !void {
    _ = try params.read(std.mem.sliceAsBytes(self.weights));
}
pub fn writeParams(self: *Self, params: anytype) !void {
    _ = try params.writeAll(std.mem.sliceAsBytes(self.weights));
}
pub fn init(
    alloc: std.mem.Allocator,
    lcommon: struct {
        batchSize: usize,
        inputSize: usize,
    },
    outputSize: usize,
) !Self {
    const inputSize = lcommon.inputSize;
    const batchSize = lcommon.batchSize;
    std.debug.assert(inputSize != 0);
    std.debug.assert(outputSize != 0);
    std.debug.assert(batchSize != 0);
    var weights: []f64 = try alloc.alloc(f64, inputSize * outputSize);
    var prng = std.Random.DefaultPrng.init(123);
    var w: usize = 0;
    while (w < inputSize * outputSize) : (w += 1) {
        weights[w] = prng.random().floatNorm(f64) * 0.2;
    }
    return Self{
        .weights = weights,
        .last_inputs = undefined,
        .fwd_out = try alloc.alloc(f64, outputSize * batchSize),
        .weight_grads = try alloc.alloc(f64, inputSize * outputSize),
        .bkw_out = try alloc.alloc(f64, inputSize * batchSize),
        .batchSize = batchSize,
        .outputSize = outputSize,
        .inputSize = inputSize,
    };
}

pub fn deinitBackwards(self: *Self, alloc: std.mem.Allocator) void {
    //alloc.free(self.last_inputs);
    //alloc.free(self.outputs);
    alloc.free(self.weight_grads);
    alloc.free(self.bkw_out);
}

pub fn forward(
    self: *Self,
    inputs: []const f64,
) void {
    std.debug.assert(inputs.len == self.inputSize * self.batchSize);
    var b: usize = 0;
    while (b < self.batchSize) : (b += 1) {
        var o: usize = 0;
        while (o < self.outputSize) : (o += 1) {
            var sum: f64 = 0;
            var i: usize = 0;
            while (i < self.inputSize) : (i += 1) {
                sum += inputs[b * self.inputSize + i] * self.weights[self.outputSize * i + o];
            }
            self.fwd_out[b * self.outputSize + o] = sum;
        }
    }
    self.last_inputs = inputs;
}

pub fn backwards(
    self: *Self,
    grads: []f64,
) void {
    std.debug.assert(self.last_inputs.len == self.inputSize * self.batchSize);

    //self.input_grads = [1]f64{0} ** (inputSize * batchSize);
    //self.weight_grads = [1]f64{0} ** (inputSize * outputSize);

    @memset(self.bkw_out, 0);
    @memset(self.weight_grads, 0);

    var b: usize = 0;
    while (b < self.batchSize) : (b += 1) {
        var i: usize = 0;
        while (i < self.inputSize) : (i += 1) {
            var o: usize = 0;
            while (o < self.outputSize) : (o += 1) {
                self.weight_grads[i * self.outputSize + o] +=
                    (grads[b * self.outputSize + o] * self.last_inputs[b * self.inputSize + i]) / @as(f64, @floatFromInt(self.batchSize));
                self.bkw_out[b * self.inputSize + i] +=
                    grads[b * self.outputSize + o] * self.weights[i * self.outputSize + o];
            }
        }
    }
}

pub fn applyGradients(self: *Self, config: anytype) void {
    const lambda = config.lambda;
    const lr = config.lr;

    var i: usize = 0;
    while (i < self.inputSize * self.outputSize) : (i += 1) {
        self.weights[i] -= lr * (self.weight_grads[i] - lambda);
    }
}
