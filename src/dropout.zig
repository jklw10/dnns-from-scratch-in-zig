const std = @import("std");

last_inputs: []const f64 = undefined,
fwd_out: []f64,
bkw_out: []f64,
dropOut: []f64,
batchSize: usize,
size: usize,
rounds: usize = roundsPerReset,

const roundsPerReset = (60000 / 100) * 2;
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
        .last_inputs = try alloc.alloc(f64, size * batchSize),
        .fwd_out = try alloc.alloc(f64, size * batchSize),
        .bkw_out = try alloc.alloc(f64, size * batchSize),
        .batchSize = batchSize,
        .size = size,
        .dropOut = try alloc.alloc(f64, size),
    };
}
pub fn deinitBackwards(self: *Self, alloc: std.mem.Allocator) void {
    alloc.free(self.bkw_out);
}

var prng = std.Random.DefaultPrng.init(123);
const limit = std.math.floatMax(f64);

const dropOutRate = 0.01;
const scale = 1.0 / (1.0 - dropOutRate);

pub fn forward(self: *Self, inputs: []f64) void {
    std.debug.assert(inputs.len == self.size * self.batchSize);
    self.rounds += 1;
    if (self.rounds >= roundsPerReset) {
        for (0..self.size) |i| {
            self.dropOut[i] = @as(f64, @floatFromInt(@intFromBool(prng.random().float(f64) >= dropOutRate))) * scale;
        }
        self.rounds = 0;
    }
    for (0..self.size * self.batchSize) |i| {
        self.fwd_out[i] = inputs[i] * self.dropOut[i % self.size];
    }
    self.last_inputs = inputs;
}

pub fn backwards(self: *Self, grads: []f64) void {
    std.debug.assert(grads.len == self.size * self.batchSize);

    for (0..self.size * self.batchSize) |i| {
        self.bkw_out[i] = grads[i] * self.dropOut[i % self.size];
    }
}
