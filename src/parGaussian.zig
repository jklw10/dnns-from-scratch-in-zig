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
        .last_inputs = try alloc.alloc(f64, size * batchSize), //no alloc needed?
        .fwd_out = try alloc.alloc(f64, (size - 3) * batchSize),
        .bkw_out = try alloc.alloc(f64, size * batchSize),
        .batchSize = batchSize,
        .size = size,
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

    for (0..self.batchSize) |b| {
        for (0..(self.size - 3)) |i| {
            const ind = b * (self.size - 3) + i;

            self.fwd_out[ind] = gaussianFilter(
                @as(f64, @floatFromInt(i)),
                inputs[b * self.size + (self.size - 3)],
                inputs[b * self.size + (self.size - 2)],
                inputs[b * self.size + (self.size - 1)],
                inputs[b * self.size + i],
            );
        }
    }
    self.last_inputs = inputs;
}

pub fn backwards(self: *Self, grads: []f64) void {
    std.debug.assert(grads.len == (self.size - 3) * self.batchSize);

    for (0..self.batchSize) |b| {
        var grad1: f64 = 0;
        var grad2: f64 = 0;
        var grad3: f64 = 0;
        for (0..(self.size - 3)) |i| {
            const ind = b * (self.size - 3) + i;
            const par = gaussianFilterGradient(
                @as(f64, @floatFromInt(i)),
                self.last_inputs[b * self.size + (self.size - 3)],
                self.last_inputs[b * self.size + (self.size - 2)],
                self.last_inputs[b * self.size + (self.size - 1)],
                self.last_inputs[b * self.size + i],
            );
            grad1 += grads[ind] * par.g1 / @as(f64, @floatFromInt(self.batchSize));
            grad2 += grads[ind] * par.g2 / @as(f64, @floatFromInt(self.batchSize));
            grad3 += grads[ind] * par.g3 / @as(f64, @floatFromInt(self.batchSize));
            self.bkw_out[b * self.size + i] = grads[ind] * par.g4;
        }
        self.bkw_out[b * self.size + (self.size - 3)] = self.grad1;
        self.bkw_out[b * self.size + (self.size - 2)] = self.grad2;
        self.bkw_out[b * self.size + (self.size - 1)] = self.grad3;
    }
}
const lr = 0.001;

pub fn applyGradients(self: *Self, lambda: f64) void {
    _ = lambda;
    self.p1 -= self.grad1 * lr;
    self.p2 -= self.grad2 * lr;
    self.p3 -= self.grad3 * lr;
}
