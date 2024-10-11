const std = @import("std");

last_inputs: []const f64 = undefined,
fwd_out: []f64,
bkw_out: []f64,
batchSize: usize,
size: usize,

grad1: f64 = 0.0,
grad2: f64 = 0.0,
grad3: f64 = 0.0,
p1: f64 = 1.0, //flatness
p2: f64 = 1.0, // width of the bump
p3: f64 = 1.0, // center of the bump

var prng = std.Random.DefaultPrng.init(123);
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
        .last_inputs = try alloc.alloc(f64, (size + 3) * batchSize),
        .fwd_out = try alloc.alloc(f64, size * batchSize),
        .bkw_out = try alloc.alloc(f64, (size + 3) * batchSize),
        .batchSize = batchSize,
        .size = (size + 3),
        .p1 = 1.0 + prng.random().floatNorm(f64) * 0.1,
        .p2 = prng.random().floatNorm(f64) / @as(f64, @floatFromInt(size)),
        .p3 = 0.5 + prng.random().floatNorm(f64) * 0.1,
    };
}

pub fn deinitBackwards(self: *Self, alloc: std.mem.Allocator) void {
    alloc.free(self.bkw_out);
}

fn gaussianFilter(i: f64, p1: f64, p2: f64, p3: f64, p4: f64) f64 {
    const exp_term = @exp(-((i + p3) * p2) * ((i + p3) * p2));
    return (p1 + exp_term * (1 - p1)) * p4;
}

fn gaussianFilterGradient(i: f64, p1: f64, p2: f64, p3: f64, p4: f64) struct { g1: f64, g2: f64, g3: f64, g4: f64 } {
    const exp_term = @exp(-((i + p3) * p2) * ((i + p3) * p2));

    const d_p1 = (1 - exp_term) * p4;
    const d_p2 = -2 * (i + p3) * (i + p3) * exp_term * (1 - p1) * p4;
    const d_p3 = -2 * (i + p3) * p2 * p2 * exp_term * (1 - p1) * p4;
    const d_p4 = p1 + exp_term * (1 - p1);

    return .{ .g1 = d_p1, .g2 = d_p2, .g3 = d_p3, .g4 = d_p4 };
}

pub fn forward(self: *Self, inputs: []f64) void {
    std.debug.assert(inputs.len == self.size * self.batchSize);

    var i: usize = 3;
    while (i < inputs.len) : (i += 1) {
        self.fwd_out[i] = gaussianFilter(@as(f64, @floatFromInt(i)), inputs[0], inputs[1], inputs[2], inputs[i]);
    }
    self.last_inputs = inputs;
}

pub fn backwards(self: *Self, grads: []f64) void {
    std.debug.assert(grads.len == self.size * self.batchSize);
    self.grad1 = 0;
    self.grad2 = 0;
    self.grad3 = 0;
    var i: usize = 3;
    while (i < self.last_inputs.len) : (i += 1) {
        const par = gaussianFilterGradient(@as(f64, @floatFromInt(i)), self.last_inputs[0], self.last_inputs[1], self.last_inputs[2], self.last_inputs[i]);
        self.grad1 += grads[i - 3] * par.g1 / @as(f64, @floatFromInt(self.batchSize));
        self.grad2 += grads[i - 3] * par.g2 / @as(f64, @floatFromInt(self.batchSize));
        self.grad3 += grads[i - 3] * par.g3 / @as(f64, @floatFromInt(self.batchSize));
        self.bkw_out[i] = grads[i] * par.g4;
    }
    self.bkw_out[0] = self.grad1;
    self.bkw_out[1] = self.grad2;
    self.bkw_out[2] = self.grad3;
}
const lr = 0.001;

pub fn applyGradients(self: *Self, lambda: f64) void {
    _ = lambda;
    self.p1 -= self.grad1 * lr;
    self.p2 -= self.grad2 * lr;
    self.p3 -= self.grad3 * lr;
}
