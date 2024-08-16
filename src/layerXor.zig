const std = @import("std");

weights: []u64,
last_inputs: []const u64,
outputs: []u64,
weight_grads: []u64, // = [1]f64{0} ** (inputSize * outputSize);
input_grads: []u64, //= [1]f64{0} ** (batchSize * inputSize);
batchSize: usize,
inputSize: usize,
outputSize: usize,

const Self = @This();
//var outputs: [batchSize * outputSize]f64 = [1]f64{0} ** (batchSize * outputSize);
pub fn setWeights(self: *Self, weights: []u64) void {
    self.weights = weights;
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
        .outputs = try alloc.alloc(u64, outputSize * batchSize),
        .weight_grads = try alloc.alloc(u64, inputSize * outputSize),
        .input_grads = try alloc.alloc(u64, inputSize * batchSize),
        .batchSize = batchSize,
        .outputSize = outputSize,
        .inputSize = inputSize,
    };
}

pub fn deinitBackwards(self: *Self, alloc: std.mem.Allocator) void {
    //alloc.free(self.last_inputs);
    //alloc.free(self.outputs);
    alloc.free(self.weight_grads);
    alloc.free(self.input_grads);
}
fn circularShiftLeft(arr: []const u64, shift: usize) u64 {
    const numBits: u6 = 63;

    // Calculate the index in the array and the bit shift within that u64
    const arrayIndex: usize = @as(usize, @intCast(shift / numBits));
    const bitShift: u6 = @as(u6, @intCast(shift % numBits));

    // Perform the shift on the relevant u64
    var result: u64 = (arr[arrayIndex] << bitShift);

    // Add bits from the next element, with wrapping if necessary
    if (arrayIndex + 1 < arr.len) {
        result |= (arr[arrayIndex + 1] >> (numBits - bitShift));
    } else {
        // Wrap around to the first element if at the end of the array
        result |= (arr[0] >> (numBits - bitShift));
    }

    return result;
}

fn circularShiftRight(arr: []const u64, shift: usize) u64 {
    const numBits: u8 = 64;

    // Calculate the index in the array and the bit shift within that u64
    const arrayIndex: usize = @as(usize, @intCast(shift / numBits));
    const bitShift: u8 = shift % numBits;

    // Perform the shift on the relevant u64
    var result: u64 = (arr[arrayIndex] >> bitShift);

    // Add bits from the previous element, with wrapping if necessary
    if (bitShift > 0) {
        if (arrayIndex > 0) {
            result |= (arr[arrayIndex - 1] << (numBits - bitShift));
        } else {
            // Wrap around to the last element if at the start of the array
            result |= (arr[arr.len - 1] << (numBits - bitShift));
        }
    }

    return result;
}
pub fn forward(self: *Self, inputs: []const u64) void {
    //if (inputs.len != self.inputSize * self.batchSize)
    //    std.debug.print("asd: {any}, {any}, {any}", .{ inputs.len, self.inputSize, self.batchSize });

    //std.debug.assert(inputs.len == self.inputSize * self.batchSize);
    var b: usize = 0;
    while (b < self.batchSize) : (b += 1) {
        var o: usize = 0;
        while (o < self.outputSize) : (o += 1) {
            var result: u64 = 0;
            var i: usize = 0;
            while (i < self.inputSize) : (i += 1) {
                result ^= circularShiftLeft(inputs, self.outputSize * i + o) ^ self.weights[self.outputSize * i + o];
            }
            self.outputs[b * self.outputSize + o] = result;
        }
    }
    self.last_inputs = inputs;
}

pub fn backwards(self: *Self, wanted_outputs: []u64) void {
    std.debug.assert(wanted_outputs.len == self.outputSize * self.batchSize);

    @memset(self.input_grads, 0);
    @memset(self.weight_grads, 0);
    for (0..self.inputSize) |i| {
        for (0..self.outputSize) |o| {
            for (0..self.batchSize) |b| {
                // Calculate the error (XOR between wanted output and current output)
                const err = wanted_outputs[b * self.outputSize + o] ^ self.outputs[b * self.outputSize + o];

                // Calculate the XOR gradient
                const xor_grad = err ^ self.weights[i * self.outputSize + o];

                // Update weight gradients
                self.weight_grads[i * self.outputSize + o] &= xor_grad;

                const nw = self.weights[i * self.outputSize + o] ^ self.weight_grads[i * self.outputSize + o];
                //TODO: check the shift correctness and step reverse correctness, try to AND batches together.

                self.input_grads[b * self.inputSize + i] ^= circularShiftLeft(self.last_inputs, self.outputSize * i + o) ^ nw;
            }
        }
    }
}

pub fn applyGradients(self: *Self) void {
    var i: usize = 0;
    while (i < self.inputSize * self.outputSize) : (i += 1) {
        self.weights[i] ^= self.weight_grads[i];
    }
}
