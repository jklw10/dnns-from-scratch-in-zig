__kernel void square_array(__global int* input_array, __global int* output_array) {
    int i = get_global_id(0);
    int value = input_array[i];
    output_array[i] = value * value;
}