const std = @import("std");

dropOut: []bool,
weights: []f64,
biases: []f64,
last_inputs: []const f64,
outputs: []f64,
weight_grads: []f64,
average_weight_gradient: []f64,
bias_grads: []f64,
input_grads: []f64,
batchSize: usize,
inputSize: usize,
outputSize: usize,
nodrop: f64 = 1.0,

rounds: f64,
const Self = @This();
const dropOutRate = 0.01;

const scale = 1.0 / (1.0 - dropOutRate);

pub fn setWeights(self: *Self, weights: []f64) void {
    self.weights = weights;
}

pub fn setBiases(self: *Self, biases: []f64) void {
    self.biases = biases;
}
pub fn readParams(self: *Self, params: anytype) !void {
    _ = try params.read(std.mem.sliceAsBytes(self.weights));
    _ = try params.read(std.mem.sliceAsBytes(self.biases));
    _ = try params.read(std.mem.sliceAsBytes(self.average_weight_gradient));
}
pub fn writeParams(self: *Self, params: anytype) !void {
    _ = try params.writeAll(std.mem.sliceAsBytes(self.weights));
    _ = try params.writeAll(std.mem.sliceAsBytes(self.biases));
    _ = try params.writeAll(std.mem.sliceAsBytes(self.average_weight_gradient));
}
var prng = std.Random.DefaultPrng.init(123);
pub fn init(
    alloc: std.mem.Allocator,
    batchSize: usize,
    inputSize: usize,
    outputSize: usize,
) !Self {
    std.debug.assert(inputSize != 0);
    std.debug.assert(outputSize != 0);
    std.debug.assert(batchSize != 0);
    var weights: []f64 = try alloc.alloc(f64, inputSize * outputSize);
    var biases: []f64 = try alloc.alloc(f64, outputSize);

    for (0..inputSize * outputSize) |w| {
        weights[w] = prng.random().floatNorm(f64) * 0.2;
    }
    for (0..outputSize) |b| {
        biases[b] = prng.random().floatNorm(f64) * 0.2;
    }
    var wg = try alloc.alloc(f64, inputSize * outputSize);
    for (0..inputSize * outputSize) |b| {
        wg[b] = prng.random().floatNorm(f64) * 0.2;
    }

    @memset(wg, 1);
    return Self{
        .dropOut = try alloc.alloc(bool, inputSize * outputSize),
        .weights = weights,
        .biases = biases,
        .last_inputs = undefined,
        .outputs = try alloc.alloc(f64, outputSize * batchSize),
        .weight_grads = try alloc.alloc(f64, inputSize * outputSize),
        .average_weight_gradient = wg,
        .bias_grads = try alloc.alloc(f64, outputSize),
        .input_grads = try alloc.alloc(f64, inputSize * batchSize),
        .batchSize = batchSize,
        .outputSize = outputSize,
        .inputSize = inputSize,
        .rounds = 0,
        .nodrop = 1.0,
    };
}

pub fn deinitBackwards(self: *Self, alloc: std.mem.Allocator) void {

    //alloc.free(self.last_inputs);
    //alloc.free(self.outputs);
    self.nodrop = scale;
    alloc.free(self.average_weight_gradient);
    alloc.free(self.weight_grads);
    alloc.free(self.bias_grads);
    alloc.free(self.input_grads);
}

pub fn forward(
    self: *Self,
    inputs: []const f64,
) void {
    if (inputs.len != self.inputSize * self.batchSize) {
        std.debug.print("size mismatch {any}, vs expected {any} * {any} = {any}", .{ inputs.len, self.inputSize, self.batchSize, self.inputSize * self.batchSize });
    }
    std.debug.assert(inputs.len == self.inputSize * self.batchSize);

    for (0..self.inputSize) |i| {
        for (0..self.outputSize) |o| {
            self.dropOut[self.outputSize * i + o] = prng.random().float(f64) < dropOutRate;
        }
    }

    var b: usize = 0;
    while (b < self.batchSize) : (b += 1) {
        var o: usize = 0;
        while (o < self.outputSize) : (o += 1) {
            var sum: f64 = 0;
            var i: usize = 0;
            while (i < self.inputSize) : (i += 1) {
                sum += inputs[b * self.inputSize + i] * self.weights[self.outputSize * i + o] * @as(f64, @floatFromInt(@intFromBool(self.dropOut[self.outputSize * i + o]))) * self.nodrop;
            }
            self.outputs[b * self.outputSize + o] = sum + self.biases[o];
        }
    }
    self.last_inputs = inputs;
}

const smoothing = 0.1;
pub fn backwards(
    self: *Self,
    grads: []f64,
) void {
    std.debug.assert(self.last_inputs.len == self.inputSize * self.batchSize);

    @memset(self.input_grads, 0);
    @memset(self.weight_grads, 0);
    @memset(self.bias_grads, 0);

    var b: usize = 0;
    while (b < self.batchSize) : (b += 1) {
        var o: usize = 0;
        while (o < self.outputSize) : (o += 1) {
            self.bias_grads[o] += grads[b * self.outputSize + o] / @as(f64, @floatFromInt(self.batchSize));
            var i: usize = 0;
            while (i < self.inputSize) : (i += 1) {
                const drop = @as(f64, @floatFromInt(@intFromBool(self.dropOut[self.outputSize * i + o])));
                const w = (grads[b * self.outputSize + o] * self.last_inputs[b * self.inputSize + i]) * drop;

                const aw = self.average_weight_gradient[i * self.outputSize + o];
                self.average_weight_gradient[i * self.outputSize + o] = aw + (smoothing * (w - aw));

                //const aw = self.average_weight_gradient[i * self.outputSize + o];
                //const wdiff = w / std.math.sign(aw) * @max(0.00001, @abs(aw));
                //const wadj = std.math.sign(wdiff) * std.math.pow(f64, @abs(wdiff), 1.5);
                self.weight_grads[i * self.outputSize + o] += (w / @as(f64, @floatFromInt(self.batchSize))) * drop;
                self.input_grads[b * self.inputSize + i] +=
                    grads[b * self.outputSize + o] * self.weights[i * self.outputSize + o] * drop;
            }
        }
    }
}

const Stat = struct {
    range: f64,
    avg: f64,
};

fn stats(arr: []f64) Stat {
    var min: f64 = std.math.floatMax(f64);
    var max: f64 = -min;
    var sum: f64 = 0.000000001;
    for (arr) |elem| {
        if (min > elem) min = elem;
        if (max < elem) max = elem;
        sum += elem;
    }
    return Stat{
        .range = @max(0.000000001, @abs(max - min)),
        .avg = sum / @as(f64, @floatFromInt(arr.len)),
    };
}
const wy = 2;
const wb = 1;
fn adjWeights(arr: []f64) []f64 {
    const gv = stats(arr);
    for (0..arr.len) |i| {
        arr[i] = ((arr[i] - gv.avg) / gv.range);
        //arr[i] = (std.math.sign(v) + 0.000000001) * @max(0.01, @abs(v));
    }
    return arr;
}

const avgPriority = 10;
//todo gradient average replaced with just .5 to see if it seems similar with regression values.

pub fn applyGradients(self: *Self) void {
    var i: usize = 0;
    while (i < self.inputSize * self.outputSize) : (i += 1) {
        const w = self.weight_grads[i];
        const wa = (std.math.sign(self.average_weight_gradient[i]) + 0.000001) * @max(0.001, @abs(self.average_weight_gradient[i]));
        const f = (w - wa) / wa;
        const c = avgPriority * @max(0.001, @abs(wa));
        const p = 1 / (1 / c + f) / c;

        _ = p;
        self.weights[i] -= 0.01 * w; //* p;

        //self.weights[i] -= 0.0000001 * std.math.sign(self.weights[i]) * @abs(self.weights[i] * self.weights[i]);
        //if (@abs(self.weights[i]) < 0.0000001) {
        //    self.weights[i] = prng.random().floatNorm(f64) * 0.2;
        //    self.average_weight_gradient[i] = prng.random().floatNorm(f64) * 0.2;
        //}
    }
    self.weights = adjWeights(self.weights);

    var o: usize = 0;
    while (o < self.outputSize) : (o += 1) {
        self.biases[o] -= 0.01 * self.bias_grads[o];
    }
    self.biases = adjWeights(self.biases);
}
