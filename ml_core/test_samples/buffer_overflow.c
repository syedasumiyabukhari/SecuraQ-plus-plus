/* Buffer Overflow (CWE-121) - stack buffer overflow via unbounded copy */
void process_network_data(char *input_buf, int buf_len) {
    char local_buf[64];
    char temp_buf[32];
    int i;
    int status = 0;

    /* Unbounded copy - input_buf may exceed 64 bytes */
    strcpy(local_buf, input_buf);

    /* Copy with wrong size - temp_buf is only 32 bytes */
    memcpy(temp_buf, input_buf, buf_len);

    /* Unsafe concatenation into already-used buffer */
    strcat(local_buf, temp_buf);

    /* Array with user-controlled index */
    int table[16];
    for (i = 0; i < 16; i++) {
        table[i] = 0;
    }
    int idx = input_buf[0];
    table[idx] = 1;

    /* Unbounded gets */
    gets(temp_buf);

    if (table[0] == 1) {
        status = 1;
    }
}
