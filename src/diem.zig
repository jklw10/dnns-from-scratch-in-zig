const std = @import("std");

const GiveLoss = true;

loss: []f64, // Array to store the loss values for each batch
input_grads: []f64, // Array to store gradients for each batch and input
comptime batchSize: usize = 100,
comptime inputSize: usize = 10,

const Self = @This();

pub fn init(
    comptime inputSize: usize,
    comptime batchSize: usize,
    alloc: std.mem.Allocator,
) !Self {
    return Self{
        .loss = try alloc.alloc(f64, batchSize),
        .input_grads = try alloc.alloc(f64, batchSize * inputSize),
        .batchSize = batchSize,
        .inputSize = inputSize,
    };
}

pub fn getLoss(self: *Self, inputs: []f64, targets: []u8, weights: [][]f64, lambda: f64) !void {
    var l2_sum: f64 = 0.0;
    for (weights) |row| {
        for (row) |weight| {
            l2_sum += weight * weight;
        }
    }
    const l2_term = (lambda / 2.0) * l2_sum;

    var euclidean_distance_sum: f64 = 0.0;
    var expected_distance: f64 = 0.0;
    var variance: f64 = 0.0;

    // Calculate Euclidean distance sum and the expected Euclidean distance
    var maxInput: f64 = -std.math.inf(f64);
    var sum: f64 = 0;
    var vm: f64 = std.math.floatMax(f64);
    var vM: f64 = -vm;
    var bd = [_]f64{0} ** self.batchSize;
    for (0..self.batchSize) |b| {
        var tvec = [_]f64{0} ** self.inputSize;
        tvec[targets[b]] = 1;
        // Find the maximum value in the inputs for the current batch
        for (0..self.inputSize) |i| {
            maxInput = @max(maxInput, inputs[b * self.inputSize + i]);
        }
        for (0..self.inputSize) |i| {
            const diff = inputs[b * self.inputSize + i] - tvec[i];
            sum += diff * diff;
        }
        bd[b] = std.math.sqrt(sum);
        vm = @min(vm, bd[b]);
        vM = @max(vM, bd[b]);
        euclidean_distance_sum += bd[b];
        expected_distance += sum;
    }
    const bs = @as(f64, @floatFromInt(self.batchSize));
    expected_distance /= bs;
    variance = std.math.pow(f64, euclidean_distance_sum / bs, 2);

    // Calculate the DIEM loss and gradients
    for (0..self.batchSize) |b| {
        var tvec = [_]f64{0} ** self.inputSize;
        tvec[targets[b]] = 1;
        const euclidean_distance = bd[b];

        const diem_loss = (vM - vm) / (variance * variance) * (euclidean_distance - expected_distance);

        if (GiveLoss) {
            self.loss[b] = diem_loss;
        }

        for (0..self.inputSize) |i| {
            var diff = inputs[b * self.inputSize + i];
            if (targets[b] == i) {
                diff -= 1;
            }
            self.input_grads[b * self.inputSize + i] = std.math.exp(diff - maxInput) / sum + l2_term;

            if (targets[b] == i) {
                self.input_grads[i] -= 1;
            }
        }
    }
}
