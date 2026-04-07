#pragma once
#ifndef TIOCGWINSZ
struct winsize { unsigned short ws_row, ws_col, ws_xpixel, ws_ypixel; };
#define TIOCGWINSZ 0x5413
static inline int ioctl(int, unsigned long, ...) { return -1; }
#endif
