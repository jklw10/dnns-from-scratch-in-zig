const std = @import("std");
const info = std.log.info;

const c = @cImport({
    @cDefine("CL_TARGET_OPENCL_VERSION", "110");
    @cInclude("CL/cl.h");
});

device: c.cl_device_id,
kernel: c.cl_kernel,
input_buffer: c.cl_mem,
output_buffer: c.cl_mem,
command_queue: c.cl_command_queue,

const Self = @This();

const CLError = error{
    GetPlatformsFailed,
    GetPlatformInfoFailed,
    NoPlatformsFound,
    GetDevicesFailed,
    GetDeviceInfoFailed,
    NoDevicesFound,
    CreateContextFailed,
    CreateCommandQueueFailed,
    CreateProgramFailed,
    BuildProgramFailed,
    CreateKernelFailed,
    SetKernelArgFailed,
    EnqueueNDRangeKernel,
    CreateBufferFailed,
    EnqueueWriteBufferFailed,
    EnqueueReadBufferFailed,
};

pub fn init() CLError!Self {
    var returned: Self = undefined;
    var platform_ids: [16]c.cl_platform_id = undefined;
    var platform_count: c.cl_uint = undefined;
    if (c.clGetPlatformIDs(platform_ids.len, &platform_ids, &platform_count) != c.CL_SUCCESS) {
        return CLError.GetPlatformsFailed;
    }
    info("{} cl platform(s) found:", .{@as(u32, @intCast(platform_count))});

    for (platform_ids[0..platform_count], 0..) |id, i| {
        var name: [1024]u8 = undefined;
        var name_len: usize = undefined;
        if (c.clGetPlatformInfo(id, c.CL_PLATFORM_NAME, name.len, &name, &name_len) != c.CL_SUCCESS) {
            return CLError.GetPlatformInfoFailed;
        }
        info("  platform {}: {s}", .{ i, name[0..name_len] });
    }

    if (platform_count == 0) {
        return CLError.NoPlatformsFound;
    }

    info("choosing platform 0...", .{});

    var device_ids: [16]c.cl_device_id = undefined;
    var device_count: c.cl_uint = undefined;
    if (c.clGetDeviceIDs(platform_ids[0], c.CL_DEVICE_TYPE_ALL, device_ids.len, &device_ids, &device_count) != c.CL_SUCCESS) {
        return CLError.GetDevicesFailed;
    }
    info("{} cl device(s) found on platform 0:", .{@as(u32, @intCast(device_count))});

    for (device_ids[0..device_count], 0..) |id, i| {
        var name: [1024]u8 = undefined;
        var name_len: usize = undefined;
        if (c.clGetDeviceInfo(id, c.CL_DEVICE_NAME, name.len, &name, &name_len) != c.CL_SUCCESS) {
            return CLError.GetDeviceInfoFailed;
        }
        info("  device {}: {s}", .{ i, name[0..name_len] });
    }

    if (device_count == 0) {
        return CLError.NoDevicesFound;
    }

    info("choosing device 0...", .{});

    returned.device = device_ids[0];
    const program_src = @embedFile("kernels/test.cl");
    info("** running test **", .{});

    const ctx = c.clCreateContext(null, 1, &returned.device, null, null, null); // future: last arg is error code
    if (ctx == null) {
        return CLError.CreateContextFailed;
    }
    defer _ = c.clReleaseContext(ctx);

    returned.command_queue = c.clCreateCommandQueue(ctx, returned.device, 0, null); // future: last arg is error code
    if (returned.command_queue == null) {
        return CLError.CreateCommandQueueFailed;
    }
    defer {
        _ = c.clFlush(returned.command_queue);
        _ = c.clFinish(returned.command_queue);
        _ = c.clReleaseCommandQueue(returned.command_queue);
    }

    var program_src_c: [*c]const u8 = program_src;
    const program = c.clCreateProgramWithSource(ctx, 1, &program_src_c, null, null); // future: last arg is error code
    if (program == null) {
        return CLError.CreateProgramFailed;
    }
    defer _ = c.clReleaseProgram(program);

    if (c.clBuildProgram(program, 1, &returned.device, null, null, null) != c.CL_SUCCESS) {
        return CLError.BuildProgramFailed;
    }

    returned.kernel = c.clCreateKernel(program, "square_array", null);
    if (returned.kernel == null) {
        return CLError.CreateKernelFailed;
    }
    defer _ = c.clReleaseKernel(returned.kernel);

    // Create buffers
    var input_array = init: {
        var init_value: [1024]i32 = undefined;
        for (0..init_value.len) |i| {
            init_value[i] = @intCast(i);
        }
        break :init init_value;
    };
    returned.input_buffer = c.clCreateBuffer(ctx, c.CL_MEM_READ_ONLY, input_array.len * @sizeOf(i32), null, null);
    if (returned.input_buffer == null) {
        return CLError.CreateBufferFailed;
    }
    defer _ = c.clReleaseMemObject(returned.input_buffer);

    returned.output_buffer = c.clCreateBuffer(ctx, c.CL_MEM_WRITE_ONLY, input_array.len * @sizeOf(i32), null, null);
    if (returned.output_buffer == null) {
        return CLError.CreateBufferFailed;
    }
    defer _ = c.clReleaseMemObject(returned.output_buffer);

    // Fill input buffer
    if (c.clEnqueueWriteBuffer(returned.command_queue, returned.input_buffer, c.CL_TRUE, 0, input_array.len * @sizeOf(i32), &input_array, 0, null, null) != c.CL_SUCCESS) {
        return CLError.EnqueueWriteBufferFailed;
    }

    return returned;
}

pub fn run(self: *Self) CLError!void {
    //const device = self.device;
    //const kernel = self.kernel;

    // Execute kernel
    if (c.clSetKernelArg(self.kernel, 0, @sizeOf(c.cl_mem), self.input_buffer) != c.CL_SUCCESS) {
        return CLError.SetKernelArgFailed;
    }
    if (c.clSetKernelArg(self.kernel, 1, @sizeOf(c.cl_mem), self.output_buffer) != c.CL_SUCCESS) {
        return CLError.SetKernelArgFailed;
    }
    const global_item_size: usize = 1024;
    var local_item_size: usize = 64;
    if (c.clEnqueueNDRangeKernel(self.command_queue, self.kernel, 1, null, &global_item_size, &local_item_size, 0, null, null) != c.CL_SUCCESS) {
        return CLError.EnqueueNDRangeKernel;
    }

    var output_array: [1024]i32 = undefined;
    if (c.clEnqueueReadBuffer(self.command_queue, self.output_buffer, c.CL_TRUE, 0, output_array.len * @sizeOf(i32), &output_array, 0, null, null) != c.CL_SUCCESS) {
        return CLError.EnqueueReadBufferFailed;
    }

    info("** done **", .{});

    info("** results **", .{});

    for (output_array, 0..) |val, i| {
        if (i % 100 == 0) {
            info("{} ^ 2 = {}", .{ i, val });
        }
    }

    info("** done, exiting **", .{});
}
