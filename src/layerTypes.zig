const std = @import("std");

const layer = @import("layer.zig");
const layerB = @import("layerBias.zig");
const layerG = @import("layerGrok.zig");
const dataSet = @import("dataSet.zig");
const relu = @import("relu.zig");
const reloid = @import("reloid.zig");
const pyramid = @import("pyramid.zig");
const gaussian = @import("gaussian.zig");
const pGaussian = @import("parGaussian.zig");
const dropout = @import("dropout.zig");
const selfPredict = @import("selfPredict.zig");

const utils = @import("utils.zig");

pub const uLayer = union(enum) {
    LayerG: usize,
    LayerB: usize,
    Layer: usize,
    Relu: void,
    Pyramid: void,
    Gaussian: void,
    PGaussian: void,
    Reloid: void,
    Dropout: void,

    SelfPredict: void,
    pub fn layerInit(comptime desc: uLayer, alloc: std.mem.Allocator, lcommon: anytype) !Layer {
        //comptime var lsize = 0;

        const lt = switch (desc) {
            .Layer => layer,
            .LayerB => layerB,
            .LayerG => layerG,
            .Reloid => reloid,
            .Relu => relu,
            .Gaussian => gaussian,
            .PGaussian => pGaussian,
            .Pyramid => pyramid,
            .Dropout => dropout,
            .SelfPredict => selfPredict,
        };
        const lconf = switch (desc) {
            inline else => |s| s,
        };
        const ltt = switch (lt) {
            inline else => |*l| try l.init(
                alloc,
                lcommon,
                lconf,
            ),
        };

        const layerType = switch (desc) {
            .Layer => Layer{ .Layer = ltt },
            .LayerB => Layer{ .LayerB = ltt },
            .LayerG => Layer{ .LayerG = ltt },
            .Relu => Layer{ .Relu = ltt },
            .Reloid => Layer{ .Reloid = ltt },
            .Pyramid => Layer{ .Pyramid = ltt },
            .Gaussian => Layer{ .Gaussian = ltt },
            .PGaussian => Layer{ .PGaussian = ltt },
            .Dropout => Layer{ .Dropout = ltt },
            .SelfPredict => Layer{ .SelfPredict = ltt },
        };

        return layerType;
    }
};

pub const Layer = union(enum) {
    LayerG: layerG,
    LayerB: layerB,
    Layer: layer,

    Relu: relu,
    Pyramid: pyramid,
    Gaussian: gaussian,
    PGaussian: pGaussian,
    Reloid: reloid,
    Dropout: dropout,
    SelfPredict: selfPredict,

    pub fn forward(this: *@This(), args: anytype) void {
        switch (this.*) {
            inline else => |*l| l.forward(args),
        }
    }
    pub fn backwards(this: *@This(), args: anytype) void {
        switch (this.*) {
            inline else => |*l| l.backwards(args),
        }
    }
    pub fn applyGradients(this: *@This(), args: anytype) void {
        utils.callIfCan(this, args, "applyGradients");
    }
};
