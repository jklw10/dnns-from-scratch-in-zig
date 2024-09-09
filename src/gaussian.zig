const std = @import("std");

last_inputs: []const f64 = undefined,
fwd_out: []f64,
bkw_out: []f64,
batchSize: usize,
size: usize,

const Self = @This();
pub const mu = 1.0; // center of the bump
pub const sigma = 1.0; // width of the bump
const sigma2 = sigma * sigma;
const continuepoint = 3;
const epsilon = gaussian_derivative(continuepoint);

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
    };
}

pub fn deinitBackwards(self: *Self, alloc: std.mem.Allocator) void {
    alloc.free(self.bkw_out);
}

fn gaussian(x: f64) f64 {
    const val = (1.0 / std.math.sqrt(2 * std.math.pi * sigma2));
    const gauss = val * @exp(-std.math.pow(f64, x, 2) / (2 * sigma2));
    //const gauss = (1.0 / std.math.sqrt(2 * std.math.pi * sigma2)) * @exp(-1 * std.math.pow(f64, (tx), 2) / (2 * sigma2));
    //todo gradient descend until gaussians gradient matches epsilon, and use that instead of "3"
    return gauss;
}
fn gaussian_derivative(x: f64) f64 {
    return -(x / sigma2) * gaussian(x);
}

pub fn leaky_gaussian(x: f64) f64 {
    const gauss = gaussian(x);
    if (@abs(x) < continuepoint) {
        return gauss;
    } else {
        return gaussian(continuepoint) + epsilon * (@abs(x) - continuepoint);
    }
}

pub fn leaky_gaussian_derivative(x: f64) f64 {
    const gd = gaussian_derivative(x);
    //_ = gaussian_derivative;
    //const Pi = std.math.pi;
    //const gaussian_derivative = std.math.sign(x) * -((-mu + x) / (std.math.pow(f64, sigma, 3) * @exp(std.math.pow(f64, (-mu + x), 2) / (2 * std.math.pow(f64, sigma, 2))) * @sqrt(2 * Pi)));

    if (@abs(x) < continuepoint) {
        return gd;
    } else {
        return epsilon * std.math.sign(x);
    }
}

pub fn forward(self: *Self, inputs: []f64) void {
    std.debug.assert(inputs.len == self.size * self.batchSize);

    var i: usize = 0;
    while (i < inputs.len) : (i += 1) {
        self.fwd_out[i] = leaky_gaussian(inputs[i] - mu);
    }
    self.last_inputs = inputs;
}

pub fn backwards(self: *Self, grads: []f64) void {
    std.debug.assert(grads.len == self.size * self.batchSize);
    var i: usize = 0;
    while (i < self.last_inputs.len) : (i += 1) {
        self.bkw_out[i] = grads[i] * leaky_gaussian_derivative(self.last_inputs[i] - mu);
    }
}
