const std = @import("std");

pub const primes100 = makeprimes(541);

fn graphFunc(func: anytype, allocator: std.mem.Allocator) void {
    var inputs: [200]f64 = undefined;
    var pyr = try func.init(allocator, 1, 200);
    for (inputs, 0..) |_, i| {
        inputs[i] = (-100 + @as(f64, @floatFromInt(i))) / 20;
    }

    pyr.forward(&inputs);
    pyr.backwards(&inputs);
    for (inputs, 0..) |_, i| {
        std.debug.print("{d:.4},", .{inputs[i]});
        std.debug.print("{d:.4},", .{func.fwd_out[i]});
        std.debug.print("{d:.4}\n", .{func.bkw_out[i]});
    }
}

fn makeprimes(until: u64) [100]f64 {
    var i = 2;
    var p = 1;
    var primes = [_]u16{2} ** 100;
    @setEvalBranchQuota(10000);
    while (i <= until) : (i += 1) {
        const root = @sqrt(@as(f64, @floatFromInt(i)));
        var C = 0;
        var nohit = 0;
        while (C < p) : (C += 1) {
            const number = primes[C];
            //if(luku >= Juuri)
            if (i % primes[C] == 0) {
                nohit += 1;
            }
            if (nohit == 0 and number >= root) {
                primes[p] = i;
                p += 1;
                break;
            } else if (nohit > 0) {
                break;
            }
        }
    }
    @setEvalBranchQuota(1000);
    var returned = [_]f64{0} ** 100;
    for (primes, 0..) |pr, id| {
        returned[id] = @as(f64, @floatFromInt(pr));
    }
    return returned;
}
pub const Stat = struct {
    range: f64,
    avg: f64,
    avgabs: f64,
};
pub fn stats(arr: []f64) Stat {
    var min: f64 = std.math.floatMax(f64);
    var max: f64 = -min;
    var sum: f64 = 0.000000001;
    var absum: f64 = 0.000000001;
    for (arr) |elem| {
        if (min > elem) min = elem;
        if (max < elem) max = elem;
        sum += elem;
        absum += @abs(elem);
    }
    return Stat{
        .range = @max(0.000000001, @abs(max - min)),
        .avg = sum / @as(f64, @floatFromInt(arr.len)),
        .avgabs = absum / @as(f64, @floatFromInt(arr.len)),
    };
}
pub fn callIfTypeMatch(t: anytype, args: anytype, comptime name: []const u8) void {
    if (@TypeOf(t) == @TypeOf(args) and @hasDecl(@TypeOf(args), name)) {
        @field(@TypeOf(t), name)(t, args);
    }
}
pub fn callIfCan(t: anytype, args: anytype, comptime name: []const u8) void {
    switch (t.*) {
        inline else => |*l| {
            if (@hasDecl(@TypeOf(l.*), name)) {
                @field(@TypeOf(l.*), name)(l, args);
            }
        },
    }
}
pub fn callIfCanErr(t: anytype, args: anytype, comptime name: []const u8) !void {
    switch (@TypeOf(t.*)) {
        inline else => |l| {
            if (@hasDecl(l, name)) {
                try @field(l, name)(l, args);
            }
        },
    }
}
pub fn normalize(arr: []f64, multi: f64, bias: f64, alpha: f64) []f64 {
    const gv = stats(arr);
    for (0..arr.len) |i| {
        arr[i] -= alpha * (arr[i] - (((arr[i] - gv.avg) / gv.range * multi) + bias));
    }
    return arr;
}
pub fn shufflePairedWindows(
    r: anytype,
    comptime T: type,
    comptime size: usize,
    buf: []T,
    comptime T2: type,
    comptime size2: usize,
    buf2: []T2,
) void {
    const MinInt = usize;
    if (buf.len < 2) {
        return;
    }
    // `i <= j < max <= maxInt(MinInt)`
    const max: MinInt = @intCast(buf.len / size);
    var i: MinInt = 0;
    while (i < max - 1) : (i += 1) {
        const j: MinInt = @intCast(r.random().intRangeLessThan(usize, i, max));
        std.mem.swap([size]T, buf[i * size ..][0..size], buf[j * size ..][0..size]);

        std.mem.swap([size2]T2, buf2[i * size2 ..][0..size2], buf2[j * size2 ..][0..size2]);
    }
}
