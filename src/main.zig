const layer = @import("layer.zig");
const layerB = @import("layerBias.zig");
const layerG = @import("layerGrok.zig");
const nll = @import("nll.zig");
const diem = @import("diem.zig");
const dataSet = @import("dataSet.zig");
const relu = @import("relu.zig");
const reloid = @import("reloid.zig");
const pyramid = @import("pyramid.zig");
const gaussian = @import("gaussian.zig");
const pGaussian = @import("parGaussian.zig");

const lt = @import("layerTypes.zig");
const nn = @import("nn.zig");

const utils = @import("utils.zig");
const std = @import("std");

const timer = false;

const scheduleItem = struct {
    epochs: usize,
    hLSize: usize,
};
const schedule = [_]scheduleItem{
    .{ .epochs = 100, .hLSize = 25 },
};

const resetEpOnRescale = true;
const continueFrom = 0;
const lambda = 0.1;
const m = std.math;
const regDim: f64 = m.phi;

const graphfuncs = false;
const multiIter = true;

const fileSignature = "G25RRRR_G10R.f64";

const reinit = false;

const bmphead = [_]u8{
    0x42, 0x4D, // BM
    0x7A, 0x00, 0x00, 0x00, //size in bytes (little endian)
    0x00, 0x00, //reserved
    0x00, 0x00, //reserved
    0x36, 0x00, 0x00, 0x00, //starting address of byte wheere bmp is found (little endian)
    0x28, //DIB header size
    0x00, 0x00, 0x00, 0x20, //width
    0x00, 0x00, 0x00, 0x20, //height
    0x00, 0x01, //color planes? 1?
    0x00, 0x18, //bits per pixel
    0x00, 0x00, 0x00, 0x00, //compression method
    0x00, 0x00, 0x00, 0x00, //image size, can be 0 without compression
    0x00, 0x00, 0x00, 0x13, //pixel per metre horizontal resolution
    0x00, 0x00, 0x00, 0x13, //pixel per metre vertical resolution
    0x00, 0x00, 0x00, 0x00, //number of colors in palette
    0x00, 0x00, 0x00, 0x00, //number of important colors.
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};

    const allocator = gpa.allocator();

    const dataset = try dataSet.cifar.dtype.readData(allocator);

    const filew = try std.fs.cwd().createFile(
        "data/debug.bmp",
        .{
            .read = true,
            .truncate = true,
        },
    );
    try filew.writeAll(&bmphead);
    for (0..32) |i| {
        const imgwidth = 32 * 3;
        for (0..32 * 3) |j| {
            try filew.writeAll(&[1]u8{@as(u8, @intFromFloat(256.0 * dataset.test_images[i * imgwidth ..][j]))});
        }
    }
    filew.close();

    defer dataset.deinit(allocator);

    if (graphfuncs) {
        utils.graphFunc(gaussian);
    }

    std.debug.print("Training... \n", .{});

    const t = std.time.milliTimestamp();
    //try runSchedule(1, dataset, allocator);

    const batchSize = 100;
    const writeFile = true;
    var arena = std.heap.ArenaAllocator.init(allocator);
    var arena2 = std.heap.ArenaAllocator.init(allocator);

    const default = lt.uLayer.Relu;

    const datatype = @TypeOf(dataset);

    const nntype = nn.NeuralNetwork(datatype, nll);

    const cs = schedule[0].hLSize;

    const layers = comptime [_]lt.uLayer{
        .{ .LayerB = cs }, default,
        .{ .LayerB = cs }, default,
        .{ .LayerB = cs }, default,
        .{ .LayerB = cs }, default,
        .{ .LayerB = 10 }, default,
    };
    const trainConfig = .{
        .deinitBackwards = false,
        .batchSize = batchSize,
        .lambda = lambda,
        .regDim = regDim,
    };
    const validationConfig = .{
        .deinitBackwards = true,
        .batchSize = datatype.validationSize,
        .lambda = lambda,
        .regDim = regDim,
    };
    var trainNet = try nntype.init(
        &layers,
        trainConfig,
        arena.allocator(),
    );
    var validationNet = try nntype.init(
        &layers,
        validationConfig,
        arena.allocator(),
    );

    inline for (0..schedule.len) |itera| {
        var sum: usize = 0;
        if (!(itera == 0)) {
            if (!resetEpOnRescale) {
                for (schedule[0 .. itera - 1]) |elem| {
                    sum += elem.epochs;
                }
            }
            const ps = schedule[itera].hLSize;
            const fileL = [_]lt.uLayer{
                .{ .LayerG = ps }, default,
                .{ .LayerG = ps }, default,
                .{ .LayerG = ps }, default,
                .{ .LayerG = ps }, default,
                .{ .LayerG = 10 }, default,
            };
            var other = try nntype.init(
                &fileL,
                trainConfig,
                arena2.allocator(),
            );
            try other.insert(trainNet);
            trainNet = other;
            validationNet = try nntype.init(
                &fileL,
                validationConfig,
                arena2.allocator(),
            );
            _ = arena.reset(.free_all);
            std.mem.swap(std.heap.ArenaAllocator, &arena, &arena2);
            //arena = arena2;
        }

        const epochs = schedule[itera].epochs;
        const pastEp = sum;
        for (0..epochs) |e| {
            try trainNet.epoch(
                &dataset,
                .{ .batchSize = batchSize, .epoch = pastEp + e, .lambda = lambda, .regDim = regDim },
            );
            validationNet.copyParams(trainNet);
            const b = validationNet.validate(dataset);
            std.debug.print("{any} \n", .{b});
        }
    }
    if (writeFile) {
        try trainNet.toFile();
    }

    const ct = std.time.milliTimestamp();

    std.debug.print("time total: {}ms\n", .{ct - t});
}
