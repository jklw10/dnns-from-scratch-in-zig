const std = @import("std");
const Param = struct {
    data: []f64,
    EMA: []f64,
    grad: []f64,
    moment: []f64,
    moment2: []f64,
    slowmoment: []f64,
    fn init(size: usize, alloc: std.mem.Allocator) !Param {
        return Param{
            .data = try alloc.alloc(f64, size),
            .grad = try alloc.alloc(f64, size),
            .EMA = try alloc.alloc(f64, size),
            .moment = try alloc.alloc(f64, size),
            .moment2 = try alloc.alloc(f64, size),
            .slowmoment = try alloc.alloc(f64, size),
        };
    }
    fn insert(self: *Param, other: Param) void {
        @memcpy(self.data[0..other.data.len], other.data);
        @memcpy(self.grad[0..other.grad.len], other.grad);
        @memcpy(self.EMA[0..other.EMA.len], other.EMA);
        @memcpy(self.moment[0..other.moment.len], other.moment);
        @memcpy(self.moment2[0..other.moment2.len], other.moment2);
        @memcpy(self.slowmoment[0..other.slowmoment.len], other.slowmoment);
    }
};

weights: Param,
biases: Param,
last_inputs: []const f64,
fwd_out: []f64,
bkw_out: []f64,
batchSize: usize,
inputSize: usize,
outputSize: usize,
rounds: f32 = 1,
const Self = @This();

pub fn copyParams(self: *Self, other: Self) void {
    self.weights = other.weights;
    self.biases = other.biases;
}

pub fn readParams(self: *Self, params: anytype) !void {
    _ = try params.read(std.mem.sliceAsBytes(self.weights));
    _ = try params.read(std.mem.sliceAsBytes(self.biases));
}
pub fn writeParams(self: Self, params: anytype) !void {
    _ = try params.writeAll(std.mem.sliceAsBytes(self.weights));
    _ = try params.writeAll(std.mem.sliceAsBytes(self.biases));
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
    var prng = std.Random.DefaultPrng.init(123);

    var returned = Self{
        .weights = try Param.init(inputSize * outputSize, alloc),
        .biases = try Param.init(outputSize, alloc),
        .last_inputs = undefined,
        .fwd_out = try alloc.alloc(f64, outputSize * batchSize),
        .bkw_out = try alloc.alloc(f64, inputSize * batchSize),
        .batchSize = batchSize,
        .outputSize = outputSize,
        .inputSize = inputSize,
    };
    var w: usize = 0;
    while (w < inputSize * outputSize) : (w += 1) {
        returned.weights.data[w] = prng.random().floatNorm(f64) * 0.2;
    }

    var b: usize = 0;
    while (b < outputSize) : (b += 1) {
        returned.biases.data[b] = prng.random().floatNorm(f64) * 0.2;
    }

    @memset(returned.weights.moment, 0);
    @memset(returned.weights.moment2, 0);
    @memset(returned.weights.slowmoment, 0);

    return returned;
}

pub fn deinitBackwards(self: *Self, alloc: std.mem.Allocator) void {

    //alloc.free(self.last_inputs);
    //alloc.free(self.outputs);
    alloc.free(self.weight_grads);
    alloc.free(self.bias_grads);
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

    var b: usize = 0;
    while (b < self.batchSize) : (b += 1) {
        var o: usize = 0;
        while (o < self.outputSize) : (o += 1) {
            var sum: f64 = 0;
            var i: usize = 0;
            while (i < self.inputSize) : (i += 1) {
                sum += inputs[b * self.inputSize + i] * self.weights.data[self.outputSize * i + o];
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
                self.weights.grad[i * self.outputSize + o] +=
                    (grads[b * self.outputSize + o] * self.last_inputs[b * self.inputSize + i]) / @as(f64, @floatFromInt(self.batchSize));
                self.bkw_out[b * self.inputSize + i] +=
                    grads[b * self.outputSize + o] * self.weights.data[i * self.outputSize + o];
            }
        }
    }
}

const roundsPerEp = 60000 / 100;

pub fn applyGradients(self: *Self, config: anytype) void {

    //const ep = @trunc(self.rounds / roundsPerEp) + 1;
    const lambda = config.lambda;
    const lr = config.lr;
    //_ = lambda;
    var i: usize = 0;
    while (i < self.inputSize * self.outputSize) : (i += 1) {
        const g = self.weights.grad[i];
        const w = self.weights.data[i];

        const beta1 = 0.9;
        const slowbeta = 0.9999;
        const beta2 = 0.999;
        const alpha = 5;
        // update the first moment (momentum)
        const m: f64 = beta1 * self.weights.moment[i] + (1.0 - beta1) * g;
        const sm: f64 = slowbeta * self.weights.slowmoment[i] + (1.0 - slowbeta) * g;
        // update the second moment (RMSprop)
        const v: f64 = beta2 * self.weights.moment2[i] + (1.0 - beta2) * g * g;
        // bias-correct both moments
        const m_hat: f64 = m / (1.0 - std.math.pow(f64, beta1, self.rounds));
        const v_hat: f64 = v / (1.0 - std.math.pow(f64, beta2, self.rounds));

        // update
        self.weights.moment[i] = m;
        self.weights.slowmoment[i] = sm;
        self.weights.moment2[i] = v;
        const smadj = sm;
        const sqrt_v_hat: f64 = std.math.sqrt(v_hat);
        const tmp = lr * ((m_hat + alpha * smadj) / (sqrt_v_hat + 1e-8) + lr * lambda * w);
        self.weights.data[i] -= tmp;
    }
    self.rounds += 1.0;
    var o: usize = 0;
    while (o < self.outputSize) : (o += 1) {
        self.biases.data[o] -= lr * self.biases.grad[o];
    }
}
