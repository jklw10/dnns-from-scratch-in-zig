const std = @import("std");

pub const Data = struct {
    train_images: []f64,
    train_labels: []u8,
    test_images: []f64,
    test_labels: []u8,
    trainSize: usize,
    inputSize: usize,
    outputSize: usize,
    validationSize: usize,
    const Self = @This();

    pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
        allocator.free(self.train_images);
        allocator.free(self.train_labels);
        allocator.free(self.test_images);
        allocator.free(self.test_labels);
    }
};

pub fn readData(
    train_images_path: []const u8,
    train_labels_path: []const u8,
    test_images_path: []const u8,
    test_labels_path: []const u8,
    trainSize: usize,
    inputSize: usize,
    outputSize: usize,
    validationSize: usize,
    allocator: std.mem.Allocator,
) !Data {
    const train_images_u8 = try readIdxFile(train_images_path, 16, allocator);
    defer allocator.free(train_images_u8);
    var train_images = try allocator.alloc(f64, inputSize * trainSize);
    var i: u32 = 0;
    while (i < inputSize * trainSize) : (i += 1) {
        const x: f64 = @as(f64, @floatFromInt(train_images_u8[i]));
        train_images[i] = x / 255;
    }

    const train_labels = try readIdxFile(train_labels_path, 8, allocator);

    const test_images_u8 = try readIdxFile(test_images_path, 16, allocator);
    defer allocator.free(test_images_u8);
    var test_images = try allocator.alloc(f64, inputSize * validationSize);
    i = 0;
    while (i < inputSize * validationSize) : (i += 1) {
        const x: f64 = @as(f64, @floatFromInt(test_images_u8[i]));
        test_images[i] = x / 255;
    }

    const test_labels = try readIdxFile(test_labels_path, 8, allocator);

    return Data{
        .train_images = train_images,
        .train_labels = train_labels,
        .test_images = test_images,
        .test_labels = test_labels,
        .trainSize = trainSize,
        .inputSize = inputSize,
        .outputSize = outputSize,
        .validationSize = validationSize,
    };
}

pub fn readIdxFile(path: []const u8, skip_bytes: u8, allocator: std.mem.Allocator) ![]u8 {
    const file = try std.fs.cwd().openFile(
        path,
        .{},
    );
    defer file.close();

    const reader = file.reader();
    try reader.skipBytes(skip_bytes, .{});
    const data = reader.readAllAlloc(
        allocator,
        1000000000,
    );
    return data;
}
