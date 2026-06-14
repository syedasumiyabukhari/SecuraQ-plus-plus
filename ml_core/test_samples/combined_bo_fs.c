/* Combined BO + FS Vulnerability (no UAF - no malloc/free) */
void process_and_log(char *user_msg, int mode) {
    char small_buf[32];
    char log_buf[64];
    int status = 0;

    /* BUFFER OVERFLOW: unbounded strcpy */
    strcpy(small_buf, user_msg);

    /* BUFFER OVERFLOW: strcat overflows */
    strcat(small_buf, user_msg);

    /* BUFFER OVERFLOW: memcpy with wrong size */
    memcpy(log_buf, user_msg, 512);

    /* FORMAT STRING: user input as format */
    printf(user_msg);

    /* FORMAT STRING: fprintf with user data */
    fprintf(stderr, user_msg);

    /* FORMAT STRING: sprintf with user format */
    sprintf(log_buf, user_msg);

    if (mode > 0) {
        status = 1;
    }
}
