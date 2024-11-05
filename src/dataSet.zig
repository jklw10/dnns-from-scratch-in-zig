const std = @import("std");

pub fn Data(
    comptime inputType: type,
    comptime outputType: type,
    comptime dataDefinition: type,
) type {
    return struct {
        train_images: []inputType,
        train_labels: []outputType,
        test_images: []inputType,
        test_labels: []outputType,
        trainSize: usize,
        pub const validationSize: usize = dataDefinition.validationSize;
        pub const outputSize: usize = dataDefinition.outputSize;
        pub const inputSize: usize = dataDefinition.inputSize;
        const Self = @This();

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.train_images);
            allocator.free(self.train_labels);
            allocator.free(self.test_images);
            allocator.free(self.test_labels);
        }

        pub fn readData(
            allocator: std.mem.Allocator,
        ) !Self {
            const train_images = try readIdxFile(
                inputType,
                dataDefinition.ImageFormat,
                dataDefinition.trainSize,
                dataDefinition.train_images_path,
                allocator,
            );

            const train_labels = try readIdxFile(
                outputType,
                dataDefinition.LabelFormat,
                dataDefinition.trainSize,
                dataDefinition.train_labels_path,
                allocator,
            );

            const test_images = try readIdxFile(
                inputType,
                dataDefinition.ImageFormat,
                dataDefinition.validationSize,
                dataDefinition.test_images_path,
                allocator,
            );

            const test_labels = try readIdxFile(
                outputType,
                dataDefinition.LabelFormat,
                dataDefinition.validationSize,
                dataDefinition.test_labels_path,
                allocator,
            );

            return Self{
                .train_images = train_images,
                .train_labels = train_labels,
                .test_images = test_images,
                .test_labels = test_labels,
                .trainSize = dataDefinition.trainSize,
            };
        }

        pub fn readIdxFile(
            comptime t: type,
            formatter: anytype,
            count: usize,
            path: []const u8,
            allocator: std.mem.Allocator,
        ) ![]t {
            const file = try std.fs.cwd().openFile(
                path,
                .{},
            );
            defer file.close();

            const reader = file.reader();
            try reader.skipBytes(formatter.skipBytes, .{});

            const data = try allocator.alloc(u8, formatter.size * count);
            var first = true;
            for (0..count) |i| {
                if (formatter.firstItem and first) {
                    first = false;
                    reader.readNoEof(data[i * formatter.size .. (i + 1) * formatter.size]) catch {
                        std.debug.print("fuck: {any}, should be {any}", .{ (i + 1) * formatter.size, formatter.size * count });
                        return error.eof;
                    };
                    continue;
                }
                reader.skipBytes(formatter.stride, .{}) catch {
                    std.debug.print("fuck: {any}, should be {any}", .{ (i + 1) * formatter.size, formatter.size * count });
                    return error.eof;
                };
                //reader.readUntilDelimiter(formatter.size);
                reader.readNoEof(data[i * formatter.size .. (i + 1) * formatter.size]) catch {
                    std.debug.print("fuck: {any}, should be {any}", .{ (i + 1) * formatter.size, formatter.size * count });
                    return error.eof;
                };
            }

            if (data.len != formatter.size * count) {
                std.debug.print("fuck: {any}, should be {any}", .{ data.len, formatter.size * count });
            }
            defer allocator.free(data);
            const dataOut = try allocator.alloc(t, formatter.size * count);
            formatter.format(t, dataOut, data, count);

            return dataOut;
        }
    };
}
pub const mnistFashion = struct {
    pub const train_images_path: []const u8 = "data/mnist-fashion/train-images-idx3-ubyte";
    pub const train_labels_path: []const u8 = "data/mnist-fashion/train-labels-idx1-ubyte";
    pub const test_images_path: []const u8 = "data/mnist-fashion/t10k-images-idx3-ubyte";
    pub const test_labels_path: []const u8 = "data/mnist-fashion/t10k-labels-idx1-ubyte";

    pub const inputSize = 28 * 28;
    pub const outputSize = 10;
    pub const trainSize = 60000;
    pub const validationSize = 10000;

    pub const dtype = Data(f64, u8, @This());

    pub const ImageFormat = struct {
        const size = inputSize;
        const skipBytes = 16;
        const firstItem = false;
        const stride = 0;
        pub fn format(comptime out: type, destination: []out, input: []u8, inputCount: usize) void {
            for (0..size * inputCount) |i| {
                const x: out = @as(out, @floatFromInt(input[i]));
                destination[i] = x / 255;
            }
        }
    };
    pub const LabelFormat = struct {
        const size = 1;
        const skipBytes = 8;
        const firstItem = false;
        const stride = 0;
        pub fn format(comptime out: type, destination: []out, input: []u8, inputCount: usize) void {
            _ = inputCount;
            @memcpy(destination, input);
        }
    };
};
pub const mnist = struct {
    pub const train_images_path: []const u8 = "data/mnist/train-images.idx3-ubyte";
    pub const train_labels_path: []const u8 = "data/mnist/train-labels.idx1-ubyte";
    pub const test_images_path: []const u8 = "data/mnist/t10k-images.idx3-ubyte";
    pub const test_labels_path: []const u8 = "data/mnist/t10k-labels.idx1-ubyte";

    pub const inputSize = 28 * 28;
    pub const outputSize = 10;
    pub const trainSize = 60000;
    pub const validationSize = 10000;

    pub const dtype = Data(f64, u8, @This());

    pub const ImageFormat = struct {
        const size = inputSize;
        const skipBytes = 16;
        const firstItem = false;
        const stride = 0;
        pub fn format(comptime out: type, destination: []out, input: []u8, inputCount: usize) void {
            for (0..size * inputCount) |i| {
                const x: out = @as(out, @floatFromInt(input[i]));
                destination[i] = x / 255;
            }
        }
    };
    pub const LabelFormat = struct {
        const size = 1;
        const skipBytes = 8;
        const firstItem = false;
        const stride = 0;
        pub fn format(comptime out: type, destination: []out, input: []u8, inputCount: usize) void {
            _ = inputCount;
            @memcpy(destination, input);
        }
    };
};
pub const cifar = struct {
    pub const train_images_path: []const u8 = "data/cifar-10-batches-bin/data_batch_1.bin";
    pub const train_labels_path: []const u8 = "data/cifar-10-batches-bin/data_batch_1.bin";
    pub const test_images_path: []const u8 = "data/cifar-10-batches-bin/test_batch.bin";
    pub const test_labels_path: []const u8 = "data/cifar-10-batches-bin/test_batch.bin";

    pub const inputSize = 32 * 32 * 3;
    pub const outputSize = 10;
    pub const trainSize = 10000;
    pub const validationSize = 10000;

    pub const dtype = Data(f64, u8, @This());

    pub const ImageFormat = struct {
        const size = 32 * 32 * 3;
        const skipBytes = 0;
        const firstItem = false;
        const stride = 1;
        pub fn format(comptime out: type, destination: []out, input: []u8, inputCount: usize) void {
            for (0..inputCount) |i| {
                const x: out = @as(out, @floatFromInt(input[i]));
                destination[i] = x / 255;
            }
        }
    };
    pub const LabelFormat = struct {
        const size = 1;
        const skipBytes = 0;
        const firstItem = true;
        const stride = 3072;
        pub fn format(comptime out: type, destination: []out, input: []u8, inputCount: usize) void {
            _ = inputCount;
            @memcpy(destination, input);
        }
    };
};
pub const dummy = struct {
    pub const train_images_path: []const u8 = "dummy.txt";
    pub const train_labels_path: []const u8 = "dummy.txt";
    pub const test_images_path: []const u8 = "dummy.txt";
    pub const test_labels_path: []const u8 = "dummy.txt";

    pub const inputSize = 1;
    pub const outputSize = 1;
    pub const trainSize = 1;
    pub const validationSize = 1;

    pub const dtype = Data(f64, u8, @This());

    pub const ImageFormat = struct {
        const size = 1;
        const skipBytes = 0;
        const firstItem = false;
        const stride = 0;
        pub fn format(comptime out: type, destination: []out, input: []u8, inputCount: usize) void {
            _ = input;
            _ = inputCount;
            @memset(destination, 0);
        }
    };
    pub const LabelFormat = struct {
        const size = 1;
        const skipBytes = 0;
        const firstItem = false;
        const stride = 0;
        pub fn format(comptime out: type, destination: []out, input: []u8, inputCount: usize) void {
            _ = input;
            _ = inputCount;
            @memset(destination, 0);
        }
    };
};
