const std = @import("std");

const GiveLoss = true;

loss: []f64, // = [1]f64{0} ** (batchSize);
input_grads: []f64, // = [1]f64{0} ** (batchSize * inputSize);
batchSize: usize,
inputSize: usize,

const Self = @This();

pub fn init(
    inputSize: usize,
    batchSize: usize,
    alloc: std.mem.Allocator,
) !Self {
    return Self{
        .loss = try alloc.alloc(f64, batchSize),
        .input_grads = try alloc.alloc(f64, batchSize * inputSize),
        .batchSize = batchSize,
        .inputSize = inputSize,
    };
}
pub fn nll(self: *Self, inputs: []f64, targets: []u8) !void {
    for (0..self.batchSize) |b| {
        var sum: f64 = 0;
        var maxInput: f64 = -std.math.inf(f64);
        // Find the maximum value in the inputs for the current batch
        for (0..self.inputSize) |i| {
            maxInput = @max(maxInput, inputs[b * self.inputSize + i]);
        }
        for (0..self.inputSize) |i| {
            sum += std.math.exp(inputs[b * self.inputSize + i] - maxInput);
            if (sum == std.math.inf(f64)) {
                std.debug.print("output with inf:\n {any},\n", .{
                    inputs[b * self.inputSize + i],
                });
                return error.divisionbyinf;
            }
        }
        const s = sum;
        //std.math.sign(sum + 0.0000001) *
        sum = (std.math.sign(sum) + 0.000000001) * @max(0.0000001, @abs(sum));
        if (sum >= std.math.inf(f64)) return error.divisionbyinf;
        if (std.math.sign(sum) == 0) {
            std.debug.print("sum:{}\n", .{s});
            return error.sumisnan;
        }
        if (sum == 0) {
            return error.divisionbyzero;
        }

        if (GiveLoss) {
            self.loss[b] = -1 * @log(std.math.exp(inputs[b * self.inputSize + targets[b]]) / sum);
        }
        for (0..self.inputSize) |i| {
            self.input_grads[b * self.inputSize + i] = std.math.exp(inputs[b * self.inputSize + i] - maxInput) / sum;
            if (i == targets[b]) {
                self.input_grads[b * self.inputSize + i] -= 1;
            }
        }
    }
}
