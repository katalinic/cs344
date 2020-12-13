#ifndef MATH_UTILS_H__
#define MATH_UTILS_H__

int min_div(int a, int b) {
    if (a % b == 0) {
        return a / b;
    } else {
        return a / b + 1;
    }
}

#endif
