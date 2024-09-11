const std = @import("std");

weights: []u64,
last_inputs: []const u64,
fwd_out: []u64,
weight_grads: []u64, // = [1]f64{0} ** (inputSize * outputSize);
bkw_out: []u64, //= [1]f64{0} ** (batchSize * inputSize);
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
    batchSize: usize,
    inputSize: usize,
    outputSize: usize,
) !Self {
    std.debug.assert(inputSize != 0);
    std.debug.assert(outputSize != 0);
    std.debug.assert(batchSize != 0);
    var weights: []u64 = try alloc.alloc(u64, inputSize * outputSize);
    var prng = std.Random.DefaultPrng.init(123);
    var w: usize = 0;
    while (w < inputSize * outputSize) : (w += 1) {
        weights[w] = prng.random().uintAtMost(u64, std.math.maxInt(u64));
    }
    return Self{
        .weights = weights,
        .last_inputs = undefined,
        .fwd_out = try alloc.alloc(u64, outputSize * batchSize),
        .weight_grads = try alloc.alloc(u64, inputSize * outputSize),
        .bkw_out = try alloc.alloc(u64, inputSize * batchSize),
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
pub fn forward(self: *Self, inputs: []const u64) void {
    std.debug.assert(inputs.len == self.inputSize * self.batchSize);
    var b: usize = 0;
    while (b < self.batchSize) : (b += 1) {
        var o: usize = 0;
        while (o < self.outputSize) : (o += 1) {
            var result: u64 = 0;
            var i: usize = 0;
            while (i < self.inputSize) : (i += 1) {
                result ^= inputs[b * self.inputSize + i] ^ self.weights[self.outputSize * i + o];
            }
            self.fwd_out[b * self.outputSize + o] = result;
        }
    }
    self.last_inputs = inputs;
}

pub fn backwards(self: *Self, wanted_outputs: []u64) void {
    std.debug.assert(wanted_outputs.len == self.outputSize * self.batchSize);

    var b: usize = 0;
    while (b < self.batchSize) : (b += 1) {
        var i: usize = 0;
        while (i < self.inputSize) : (i += 1) {
            var o: usize = 0;
            while (o < self.outputSize) : (o += 1) {
                // Calculate the error (XOR between wanted output and current output)
                const err = wanted_outputs[b * self.outputSize + o] ^ self.fwd_out[b * self.outputSize + o];

                // Calculate the XOR gradient
                const xor_grad = err ^ self.weights[i * self.outputSize + o];

                // Update weight gradients (optional if you need accumulation)
                self.weight_grads[i * self.outputSize + o] = xor_grad;
            }
        }
    }
}

pub fn applyGradients(self: *Self, funval: f64) void {
    _ = funval;
    // You could either apply the gradients here or directly in the backwards pass.
    var i: usize = 0;
    while (i < self.inputSize * self.outputSize) : (i += 1) {
        self.weights[i] ^= self.weight_grads[i]; // Example operation
    }
}
