const std = @import("std");
const utils = @import("utils.zig");
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
    fn insert(self: *Param, other: Param) void {
        @memcpy(self.data[0..other.data.len], other.data);
        @memcpy(self.grad[0..other.grad.len], other.grad);
        @memcpy(self.EMA[0..other.EMA.len], other.EMA);
        @memcpy(self.moment[0..other.moment.len], other.moment);
        @memcpy(self.moment2[0..other.moment2.len], other.moment2);
    }
};

dropOut: []bool,

weights: Param,
biases: Param,

maxAvgGrad: f64 = 0,
normMulti: f64 = 1,
normBias: f64 = 0,

last_inputs: []const f64,
fwd_out: []f64,
bkw_out: []f64,

batchSize: usize,
inputSize: usize,
outputSize: usize,

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
pub fn rescale(self: *Self, other: Self) void {
    self.weights.insert(other.weights);
    self.biases.insert(other.biases);
    for (other.inputSize * other.outputSize..self.inputSize * self.outputSize) |w| {
        const dev = @as(f64, @floatFromInt(self.inputSize));
        self.weights.data[w] = 0;
        self.weights.EMA[w] = prng.random().floatNorm(f64) * @sqrt(2.0 / dev);
    }
    for (other.outputSize..self.outputSize) |b| {
        self.biases.data[b] = 0;
        self.biases.EMA[b] = prng.random().floatNorm(f64) * 0.01;
    }
}

pub fn readParams(self: *Self, params: anytype) !void {
    _ = try params.read(std.mem.sliceAsBytes(self.weights.data));
    _ = try params.read(std.mem.sliceAsBytes(self.biases.data));

    _ = try params.read(std.mem.sliceAsBytes(self.weights.EMA));
    _ = try params.read(std.mem.sliceAsBytes(self.biases.EMA));

    _ = try params.read(std.mem.asBytes(&self.normMulti));
    _ = try params.read(std.mem.asBytes(&self.normBias));
    _ = try params.read(std.mem.asBytes(&self.maxAvgGrad));
}
pub fn writeParams(self: *const Self, params: anytype) !void {
    _ = try params.writeAll(std.mem.sliceAsBytes(self.weights.data));
    _ = try params.writeAll(std.mem.sliceAsBytes(self.biases.data));

    _ = try params.writeAll(std.mem.sliceAsBytes(self.weights.EMA));
    _ = try params.writeAll(std.mem.sliceAsBytes(self.biases.EMA));

    _ = try params.writeAll(std.mem.asBytes(&self.normMulti));
    _ = try params.writeAll(std.mem.asBytes(&self.normBias));
    _ = try params.writeAll(std.mem.asBytes(&self.maxAvgGrad));
}
var prng = std.Random.DefaultPrng.init(123);

pub fn reinit(self: *Self, percent: f64) void {
    _ = percent;
    //const wa = stats(self.weights);
    //std.debug.print("stats: {any}", .{wa});
    for (0..self.inputSize * self.outputSize) |w| {
        if (w > self.inputSize * self.outputSize / 4) {
            self.weights.data[w] = 0;
        }
        //const sqrI = @sqrt(2.0 / @as(f64, @floatFromInt(self.inputSize)));
        //const dev = wa.range * sqrI * percent;
        //self.weights[w] = (self.EMAWeight[w]) + prng.random().floatNorm(f64) * dev;

        //self.weights[w] = (wa.avg/ 2) + prng.random().floatNorm(f64) * dev;
        //const rand = prng.random().floatNorm(f64);
        //self.weights[w] = std.math.sign(rand) * wa.avgabs + rand * dev; //0 ring.
    }
    @memset(self.weights.data, 0);
    @memset(self.biases.data, 0);
    //@memcpy(self.averageWeights, self.weights);

    //const bs = stats(self.biases);
    //for (0..self.outputSize) |b| {
    //    self.biases[b] = self.averageBiases[b] + prng.random().floatNorm(f64) * bs.range;
    //}
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
    //std.debug.print("i:{any},o:{any},b:{any}\n", .{ inputSize, outputSize, batchSize });
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
    for (0..inputSize) |i| {
        for (0..outputSize) |o| {
            const dev = @as(f64, @floatFromInt(inputSize));
            returned.weights.data[i * outputSize + o] = prng.random().floatNorm(f64) * @sqrt(2.0 / dev);
        }
    }
    for (0..outputSize) |b| {
        returned.biases.data[b] = prng.random().floatNorm(f64) * 0.01; //good value, great value, one of the greatest.
    }

    @memcpy(returned.weights.EMA, returned.weights.data);
    @memcpy(returned.biases.EMA, returned.biases.data);

    @memcpy(returned.weights.moment, returned.weights.data);
    @memcpy(returned.biases.moment, returned.biases.data);

    //@memset(returned.weights.data, 0);
    //@memset(returned.biases.data, 0);

    //@memset(returned.biases.EMA, 0);
    //@memset(returned.weights.EMA, 0);
    //@memset(returned.biases.moment, 0);
    //@memset(returned.weights.moment, 0);

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
    for (0..self.batchSize) |b| {
        for (0..self.outputSize) |o| {
            var sum: f64 = 0;
            for (0..self.inputSize) |i| {
                const d = 1.0; // if (usedrop) 1.0 else @as(f64, @floatFromInt(@intFromBool(self.dropOut[self.outputSize * i + o]))) * self.nodrop;
                const w = funnyMulti(self.weights.data[i + self.inputSize * o], self.weights.EMA[i + self.inputSize * o]);
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
                    grads[b * self.outputSize + o] *
                    self.weights.data[i + self.inputSize * o];
                //funnyMulti(self.weights.data[i + self.inputSize * o], self.weights.EMA[i + self.inputSize * o]); //  * drop;
                //}
            }
        }
    }
}
fn funnyMulti(x: f64, y: f64) f64 {
    return std.math.sign(x) * @sqrt(@abs(x * y));
}
const roundsPerEp = 60000 / 100;
//const lr = 0.0005;
const smoothing = 0.1;

//best on its own: 0.0075;
//const lambda = 0.0075;
const elasticAlpha = 0.0;

pub fn applyGradients(self: *Self, config: anytype) void {
    self.rounds += 1.0;
    const lambda = config.lambda;
    const lr = config.lr; // / ((self.rounds / roundsPerEp) + 1);

    const normlr = lr / 10.0;

    const awstat = utils.stats(self.weights.EMA);
    const wstat = utils.stats(self.weights.data);
    const wgstat = utils.stats(self.weights.grad);

    self.normMulti -= normlr * (wgstat.range - self.normMulti);
    //self.normBias -= wgstat.avg * normlr;

    const inputFract = 2.0 / @as(f64, @floatFromInt(self.inputSize));
    _ = .{ awstat, wstat, inputFract };
    //self.weights.grad = normalize(self.weights.grad, 2 - He, 0, 1);
    const wsize = self.inputSize * self.outputSize;
    for (0..wsize) |i| {
        const wema = self.weights.EMA[i];
        const w = self.weights.data[i]; //funnyMulti(self.weights.data[i], wema);
        const fw = funnyMulti(self.weights.data[i], wema);

        //const l2 = lambda * w;

        const fractional_p = config.regDim;
        const l_p = lambda * std.math.sign(w) * std.math.pow(f64, @abs(w), fractional_p - 1);

        const abng = self.weights.grad[i];
        var g = ((abng - wgstat.avg) / wgstat.range) * (2 - inputFract);
        //_ = l2;
        g = g + l_p; // Updating the gradient using the fractional Lp regularization
        //g = g + l2; // + l2; // + EN;
        //todo, try normalize gradients.
        //weight average, use with lambda?
        //nudge towards it with \/ ?

        const awdiff = wema - fw;
        //const gdiff = 1.0 / (0.5 + @abs(g - awdiff));
        const gdiff = 1.0 / ((@abs(wema)) + @abs(g - awdiff));

        //const moment = self.weights.moment[i];
        //const mdiff = @sqrt(1.0 / (moment + @abs(@abs(abng) - moment)));
        //if (self.weights.moment[i] < 1e-3) {
        //    self.weights.moment[i] = @abs(self.weights.data[i]);
        //    self.weights.data[i] = 0;
        //    std.debug.print("reinit", .{});
        //}
        _ = .{gdiff};
        self.weights.data[i] -= lr * g * gdiff; // * mdiff;
        self.weights.EMA[i] += (smoothing * (w - wema));
        self.weights.moment[i] += 0.5 * (@abs(abng) - self.weights.moment[i]);
    }
    //1.0625
    self.weights.data = utils.normalize(self.weights.data, 1 + inputFract, 0, 1);

    //TODO: untest this:
    self.biases.grad = utils.normalize(self.biases.grad, 2 - inputFract, 0, 1);
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
        //TODO: test this:
        @memcpy(self.weights.EMA, self.weights.data);
        //@memcpy(self.weights.moment, self.weights.grad);
    } else {
        self.maxAvgGrad -= wgstat.avgabs;
    }
}
