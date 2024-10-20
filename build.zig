const builtin = @import("builtin");
const std = @import("std");

const opencl = false;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "simple-feed-forward-neural-networks-in-zig",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    if (opencl) {
        exe.addIncludePath(b.path("libs/opencl-headers"));

        if (builtin.os.tag == .windows) {
            std.debug.print("Windows detected, adding default CUDA SDK x64 lib search path. Change this in build.zig if needed...", .{});
            exe.addLibraryPath(.{ .cwd_relative = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64" });
            //TODO : amd support?
        }
        exe.linkSystemLibrary("c");

        if (builtin.os.tag == .linux) {
            exe.linkSystemLibrary("OpenCL");
        }
    }

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
