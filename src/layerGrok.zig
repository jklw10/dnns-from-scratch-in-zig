const std = @import("std");

const Self = @This();

dropOut: []bool,
weights: []f64,
biases: []f64,
last_inputs: []const f64,
outputs: []f64,
weight_grads: []f64,
averageWeights: []f64,
bias_grads: []f64,
input_grads: []f64,
batchSize: usize,
inputSize: usize,
outputSize: usize,

normMulti: f64 = 1,
normBias: f64 = 0,

nodrop: f64 = 1.0,
rounds: f64 = batchdropskip + 1,

const batchdropskip = 0.5;
const dropOutRate = 0.00;

const scale = 1.0 / (1.0 - dropOutRate);
const usedrop = false;

pub fn setWeights(self: *Self, weights: []f64) void {
    self.weights = weights;
}

pub fn setBiases(self: *Self, biases: []f64) void {
    self.biases = biases;
}
pub fn readParams(self: *Self, params: anytype) !void {
    _ = try params.read(std.mem.sliceAsBytes(self.weights));
    _ = try params.read(std.mem.sliceAsBytes(self.biases));

    _ = try params.read(std.mem.sliceAsBytes(self.averageWeights));
    _ = try params.read(std.mem.asBytes(&self.normMulti));
    _ = try params.read(std.mem.asBytes(&self.normBias));
}
pub fn writeParams(self: *Self, params: anytype) !void {
    _ = try params.writeAll(std.mem.sliceAsBytes(self.weights));
    _ = try params.writeAll(std.mem.sliceAsBytes(self.biases));

    _ = try params.writeAll(std.mem.sliceAsBytes(self.averageWeights));
    _ = try params.writeAll(std.mem.asBytes(&self.normMulti));
    _ = try params.writeAll(std.mem.asBytes(&self.normBias));
}
var prng = std.Random.DefaultPrng.init(123);

pub fn reinit(self: *Self, percent: f64) void {
    const wa = stats(self.weights);
    //const ba = stats(self.biases);
    //if (usedrop) {
    //    self.rounds += 1.0;
    //    if (self.rounds >= batchdropskip) {
    //        //std.debug.print("round 10", .{});
    //        self.rounds = 0.0; //todo move from being perbatch to per epoch, etc.
    //        for (0..self.inputSize) |i| {
    //            for (0..self.outputSize) |o| {
    //                //const d: f64 = @as(f64, @floatFromInt(self.outputSize * i + o + 1)) / @as(f64, @floatFromInt(self.outputSize * self.inputSize));
    //                self.dropOut[self.outputSize * i + o] = prng.random().float(f64) >= dropOutRate; // / (d * 2);
    //            }
    //        }
    //    }
    //}
    //std.debug.print("stats: {any}", .{wa});
    for (0..self.inputSize * self.outputSize) |w| {
        const sqrI = @sqrt(2.0 / @as(f64, @floatFromInt(self.inputSize)));
        const dev = wa.range * sqrI * percent;
        self.weights[w] = (self.averageWeights[w]) + prng.random().floatNorm(f64) * dev;

        //self.weights[w] = (wa.avg/ 2) + prng.random().floatNorm(f64) * dev;
        //const rand = prng.random().floatNorm(f64);
        //self.weights[w] = std.math.sign(rand) * wa.avgabs + rand * dev; //0 ring.

        //todo: try kill avg
    }

    //@memcpy(self.averageWeights, self.weights);
    //todo: try with and without.

    for (0..self.outputSize) |b| {
        self.biases[b] = prng.random().floatNorm(f64) * 0.2;
    }
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
    var weights: []f64 = try alloc.alloc(f64, inputSize * outputSize);
    var biases: []f64 = try alloc.alloc(f64, outputSize);

    for (0..inputSize * outputSize) |w| {
        const dev = @as(f64, @floatFromInt(inputSize));
        weights[w] = prng.random().floatNorm(f64) * @sqrt(2.0 / dev);
    }
    for (0..outputSize) |b| {
        biases[b] = prng.random().floatNorm(f64) * 0.2;
    }
    const wg = try alloc.alloc(f64, inputSize * outputSize);
    //for (0..inputSize * outputSize) |b| {
    //    wg[b] = prng.random().floatNorm(f64) * 0.2;
    //}
    @memcpy(wg, weights);
    return Self{
        .dropOut = try alloc.alloc(bool, inputSize * outputSize),
        .weights = weights,
        .biases = biases,
        .last_inputs = undefined,
        .outputs = try alloc.alloc(f64, outputSize * batchSize),
        .weight_grads = try alloc.alloc(f64, inputSize * outputSize),
        .averageWeights = wg,
        .bias_grads = try alloc.alloc(f64, outputSize),
        .input_grads = try alloc.alloc(f64, inputSize * batchSize),
        .batchSize = batchSize,
        .outputSize = outputSize,
        .inputSize = inputSize,
        .rounds = -60000 / 100,
        .nodrop = 1.0,
    };
}

pub fn deinitBackwards(self: *Self, alloc: std.mem.Allocator) void {

    //alloc.free(self.last_inputs);
    //alloc.free(self.outputs);
    self.nodrop = scale;
    alloc.free(self.averageWeights);
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

    //if (usedrop) {
    //    self.rounds += 1.0;
    //    if (self.rounds >= batchdropskip) {
    //        //std.debug.print("round 10", .{});
    //        self.rounds = 0.0; //todo move from being perbatch to per epoch, etc.
    //        for (0..self.inputSize) |i| {
    //            for (0..self.outputSize) |o| {
    //                //const d: f64 = @as(f64, @floatFromInt(self.outputSize * i + o + 1)) / @as(f64, @floatFromInt(self.outputSize * self.inputSize));
    //                self.dropOut[self.outputSize * i + o] = prng.random().float(f64) >= dropOutRate; // / (d * 2);
    //            }
    //        }
    //    }
    //}
    var b: usize = 0;
    while (b < self.batchSize) : (b += 1) {
        var o: usize = 0;
        while (o < self.outputSize) : (o += 1) {
            var sum: f64 = 0;
            var i: usize = 0;
            while (i < self.inputSize) : (i += 1) {
                const d = 1.0; // if (usedrop) 1.0 else @as(f64, @floatFromInt(@intFromBool(self.dropOut[self.outputSize * i + o]))) * self.nodrop;
                sum += d * inputs[b * self.inputSize + i] * self.weights[i + self.inputSize * o];
            }
            self.outputs[b * self.outputSize + o] = sum + self.biases[o];
        }
    }
    self.last_inputs = inputs;
}

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
                //if (!usedrop or self.dropOut[self.outputSize * i + o]) {
                //const drop = @as(f64, @floatFromInt(@intFromBool(self.dropOut[self.outputSize * i + o])));
                const w = (grads[b * self.outputSize + o] * self.last_inputs[b * self.inputSize + i]); // * drop;

                //const aw = self.average_weight_gradient[i * self.outputSize + o];
                //self.average_weight_gradient[i * self.outputSize + o] = aw + (smoothing * (w - aw));
                //
                //const wdiff = w / std.math.sign(aw) * @max(0.00001, @abs(aw));
                //const wadj = std.math.sign(wdiff) * std.math.pow(f64, @abs(wdiff), 1.5);
                //_ = wadj;
                //todo scale by variance of weight grad
                self.weight_grads[i + self.inputSize * o] += (w / @as(f64, @floatFromInt(self.batchSize))); // * wadj; //  * drop;
                self.input_grads[b * self.inputSize + i] +=
                    grads[b * self.outputSize + o] * self.weights[i + self.inputSize * o]; //  * drop;
                //}
            }
        }
    }
}

const Stat = struct {
    range: f64,
    avg: f64,
    avgabs: f64,
};

fn stats(arr: []f64) Stat {
    var min: f64 = std.math.floatMax(f64);
    var max: f64 = -min;
    var sum: f64 = 0.000000001;
    var absum: f64 = 0.000000001;
    for (arr) |elem| {
        if (min > elem) min = elem;
        if (max < elem) max = elem;
        sum += elem;
        absum += @abs(elem);
    }
    return Stat{
        .range = @max(0.000000001, @abs(max - min)),
        .avg = sum / @as(f64, @floatFromInt(arr.len)),
        .avgabs = absum / @as(f64, @floatFromInt(arr.len)),
    };
}

fn normalize(arr: []f64, multi: f64, bias: f64, alpha: f64) []f64 {
    const gv = stats(arr);
    for (0..arr.len) |i| {
        arr[i] -= alpha * (arr[i] - (((arr[i] - gv.avg) / gv.range * multi) + bias));
    }
    return arr;
}

const roundsPerEp = 60000 / 100;
const lr = 0.001;
const smoothing = lr;
//todo gradient average replaced with just .5 to see if it seems similar with regression values.
const normlr = lr / 10.0;
//best on its own: 0.0075;
const lambda = 0.0075;
const elasticAlpha = 0.0;
pub fn applyGradients(self: *Self) void {
    const wavg = stats(self.weight_grads).avg;
    const bavg = stats(self.bias_grads).avg;
    self.normMulti -= wavg * normlr;
    self.normBias -= bavg * normlr;
    //todo: check for sanity?

    self.rounds += 1.0;

    //if (self.rounds < -1 and self.rounds >= -3)
    //@memcpy(self.average_weight_gradient, self.weights);
    //if (self.rounds >= roundsPerEp * 2)
    //    self.rounds = 0.0;

    for (0..self.inputSize * self.outputSize) |i| {
        const l2 = lambda * self.weights[i];
        const l1 = lambda * std.math.sign(self.weights[i]);

        const EN = std.math.lerp(l2, l1, elasticAlpha);
        //_ = EN;
        const g = self.weight_grads[i] + EN;
        //todo, try normalize gradients.
        //weight average, use with lambda?
        //nudge towards it with \/ ?

        const awdiff = self.averageWeights[i] - self.weights[i];
        //const crossover = std.math.clamp(1.0 - ((g * lr - awdiff * avgPriority)), 0.0, 1.0);
        const gdiff = 1.0 / (0.5 + @abs(g - awdiff));
        //_ = gdiff;
        //const gdiff = 1.0 / (avgPriority + @abs(g - awdiff));
        self.weights[i] -= lr * g * gdiff; // * p; //* gadj; //* p;

        const aw = self.averageWeights[i];
        self.averageWeights[i] = aw + (smoothing * (self.weights[i] - aw));
    }

    //self.weights = normalize(self.weights, self.normMulti, self.normBias, 0.0001);

    for (0..self.outputSize) |o| {
        self.biases[o] -= lr * self.bias_grads[o]; // (self.bias_grads[o] + lambda * self.biases[o]);
    }
    //if (self.rounds >= roundsPerEp * 5) {
    //    self.rounds = 0.0;
    //    self.reinit(0.00);
    //}
    self.biases = normalize(self.biases, self.normMulti, self.normBias, 0.01);
}
