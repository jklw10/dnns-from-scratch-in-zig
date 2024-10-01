const std = @import("std");

const Self = @This();

const Param = struct {
    data: []f64,
    EMA: []f64,
    grad: []f64,
    moment: []f64,
    moment2: []f64,
    fn init(size: usize, alloc: std.mem.Allocator) !Param {
        return Param{
            .data = try alloc.alloc(f64, size),
            .grad = try alloc.alloc(f64, size),
            .EMA = try alloc.alloc(f64, size),
            .moment = try alloc.alloc(f64, size),
            .moment2 = try alloc.alloc(f64, size),
        };
    }
};

dropOut: []bool,

weights: Param,
biases: Param,

//weights: []f64,
//weight_grads: []f64,
//EMAWeight: []f64,
//moment: []f64,
//moment2: []f64,

//biases: []f64,
//bias_grads: []f64,
//averageBiases: []f64,

last_inputs: []const f64,
fwd_out: []f64,
bkw_out: []f64,

batchSize: usize,
inputSize: usize,
outputSize: usize,

maxAvgGrad: f64 = 0,
normMulti: f64 = 1,
normBias: f64 = 0,

nodrop: f64 = 1.0,
rounds: f64 = batchdropskip + 1,

const batchdropskip = 0.5;
const dropOutRate = 0.00;

const scale = 1.0 / (1.0 - dropOutRate);
const usedrop = false;

pub fn copyParams(self: *Self, other: Self) void {
    self.weights = other.weights;
    self.biases = other.biases;
}

pub fn readParams(self: *Self, params: anytype) !void {
    _ = try params.read(std.mem.sliceAsBytes(self.weights.data));
    _ = try params.read(std.mem.sliceAsBytes(self.biases.data));

    _ = try params.read(std.mem.sliceAsBytes(self.weights.EMA));
    _ = try params.read(std.mem.sliceAsBytes(self.biases.EMA));

    //_ = try params.read(std.mem.sliceAsBytes(self.EMAWeight));
    //_ = try params.read(std.mem.sliceAsBytes(self.averageBiases));

    _ = try params.read(std.mem.asBytes(&self.normMulti));
    _ = try params.read(std.mem.asBytes(&self.normBias));
    _ = try params.read(std.mem.asBytes(&self.maxAvgGrad));
}
pub fn writeParams(self: *Self, params: anytype) !void {
    _ = try params.writeAll(std.mem.sliceAsBytes(self.weights.data));
    _ = try params.writeAll(std.mem.sliceAsBytes(self.biases.data));

    _ = try params.writeAll(std.mem.sliceAsBytes(self.weights.EMA));
    _ = try params.writeAll(std.mem.sliceAsBytes(self.biases.EMA));

    //_ = try params.writeAll(std.mem.sliceAsBytes(self.EMAWeight));
    //_ = try params.writeAll(std.mem.sliceAsBytes(self.averageBiases));

    _ = try params.writeAll(std.mem.asBytes(&self.normMulti));
    _ = try params.writeAll(std.mem.asBytes(&self.normBias));
    _ = try params.writeAll(std.mem.asBytes(&self.maxAvgGrad));
}
var prng = std.Random.DefaultPrng.init(123);

pub fn reinit(self: *Self, percent: f64) void {
    const wa = stats(self.weights);
    //std.debug.print("stats: {any}", .{wa});
    for (0..self.inputSize * self.outputSize) |w| {
        const sqrI = @sqrt(2.0 / @as(f64, @floatFromInt(self.inputSize)));
        const dev = wa.range * sqrI * percent;
        self.weights[w] = (self.EMAWeight[w]) + prng.random().floatNorm(f64) * dev;

        //self.weights[w] = (wa.avg/ 2) + prng.random().floatNorm(f64) * dev;
        //const rand = prng.random().floatNorm(f64);
        //self.weights[w] = std.math.sign(rand) * wa.avgabs + rand * dev; //0 ring.
    }

    //@memcpy(self.averageWeights, self.weights);

    const bs = stats(self.biases);
    for (0..self.outputSize) |b| {
        self.biases[b] = self.averageBiases[b] + prng.random().floatNorm(f64) * bs.range;
    }
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
    std.debug.print("i:{any},o:{any},b:{any}\n", .{ inputSize, outputSize, batchSize });
    std.debug.assert(inputSize != 0);
    std.debug.assert(outputSize != 0);
    std.debug.assert(batchSize != 0);

    const returned = Self{
        .dropOut = try alloc.alloc(bool, inputSize * outputSize),
        .last_inputs = undefined,
        .fwd_out = try alloc.alloc(f64, outputSize * batchSize),
        .bkw_out = try alloc.alloc(f64, inputSize * batchSize),

        .weights = try Param.init(inputSize * outputSize, alloc),

        .biases = try Param.init(outputSize, alloc),

        .batchSize = batchSize,
        .outputSize = outputSize,
        .inputSize = inputSize,
        .rounds = -60000 / 100,
        .nodrop = 1.0,
    };
    for (0..inputSize * outputSize) |w| {
        const dev = @as(f64, @floatFromInt(inputSize));
        returned.weights.data[w] = prng.random().floatNorm(f64) * @sqrt(2.0 / dev);
    }
    for (0..outputSize) |b| {
        returned.biases.data[b] = prng.random().floatNorm(f64) * 0.01; //good value, great value, one of the greatest.
    }
    @memcpy(returned.weights.EMA, returned.weights.data);
    @memcpy(returned.biases.EMA, returned.biases.data);

    //@memset(returned.biases.EMA, 0);
    //@memset(returned.weights.EMA, 0);

    @memset(returned.biases.moment, 0);
    @memset(returned.weights.moment, 0);

    @memset(returned.biases.moment2, 0);
    @memset(returned.weights.moment2, 0);
    return returned;
}

pub fn deinitBackwards(self: *Self, alloc: std.mem.Allocator) void {

    //alloc.free(self.last_inputs);
    //alloc.free(self.outputs);
    self.nodrop = scale;
    alloc.free(self.weights.EMA);
    alloc.free(self.weights.grad);
    alloc.free(self.weights.moment);
    alloc.free(self.weights.moment2);
    alloc.free(self.biases.grad);
    alloc.free(self.biases.moment);
    alloc.free(self.biases.moment2);
    alloc.free(self.biases.EMA);
    alloc.free(self.bkw_out);
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
                const w = self.weights.data[i + self.inputSize * o];
                const in = inputs[b * self.inputSize + i];
                sum += d * in * w;
            }
            self.fwd_out[b * self.outputSize + o] = sum + self.biases.data[o];
        }
    }
    self.last_inputs = inputs;
}

pub fn backwards(
    self: *Self,
    grads: []f64,
) void {
    std.debug.assert(self.last_inputs.len == self.inputSize * self.batchSize);

    @memset(self.bkw_out, 0);
    @memset(self.weights.grad, 0);
    @memset(self.biases.grad, 0);

    var b: usize = 0;
    while (b < self.batchSize) : (b += 1) {
        var o: usize = 0;
        while (o < self.outputSize) : (o += 1) {
            self.biases.grad[o] += grads[b * self.outputSize + o] / @as(f64, @floatFromInt(self.batchSize));

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
                self.weights.grad[i + self.inputSize * o] += (w / @as(f64, @floatFromInt(self.batchSize))); // * wadj; //  * drop;
                self.bkw_out[b * self.inputSize + i] +=
                    grads[b * self.outputSize + o] * self.weights.data[i + self.inputSize * o]; //  * drop;
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
const smoothing = 0.1;
const normlr = lr / 10.0;

//best on its own: 0.0075;
//const lambda = 0.0075;
const elasticAlpha = 0.0;

pub fn applyGradients(self: *Self, lambda: f64) void {
    const awstat = stats(self.weights.EMA);
    const wstat = stats(self.weights.data);
    const wgstat = stats(self.weights.grad);

    self.normMulti -= normlr * (wgstat.range - self.normMulti);
    //self.normBias -= wgstat.avg * normlr;

    self.rounds += 1.0;

    const He = @sqrt(2.0 / @as(f64, @floatFromInt(self.inputSize)));
    _ = .{ awstat, wstat, He };
    self.weights.grad = normalize(self.weights.grad, 1, 0, 1);

    for (0..self.inputSize * self.outputSize) |i| {
        const w = self.weights.data[i];

        const l2 = lambda * w;

        var g = self.weights.grad[i];
        const wema = self.weights.EMA[i];

        //_ = l2;
        g = g + l2; // + l2; // + EN;
        //todo, try normalize gradients.
        //weight average, use with lambda?
        //nudge towards it with \/ ?

        const awdiff = wema - w;
        //const gdiff = 1.0 / (0.5 + @abs(g - awdiff));
        const gdiff = 1.0 / ((@abs(wema)) + @abs(g - awdiff));
        //_ = gdiff;
        self.weights.data[i] -= lr * g * gdiff; // * p; //* gadj; //* p;

        self.weights.EMA[i] += (smoothing * (w - wema)); //
        //const aw = self.averageWeights[i];

    }

    //self.weights.data = normalize(self.weights.data, self.normMulti, self.normBias, 1);

    for (0..self.outputSize) |o| {
        const g = self.biases.grad[o];
        const bema = self.biases.EMA[o];
        const b = self.biases.data[o];

        const abdiff = bema - b;
        const gdiff = 1.0 / (@abs(bema) + @abs(g - abdiff));

        self.biases.data[o] -= lr * g * gdiff;

        self.biases.EMA[o] += smoothing * (b - bema);
    }

    if (self.maxAvgGrad < wgstat.avgabs) {
        self.maxAvgGrad = wgstat.avgabs;
        @memcpy(self.weights.EMA, self.weights.data);
    } else {
        self.maxAvgGrad -= wgstat.avgabs;
    }
}
