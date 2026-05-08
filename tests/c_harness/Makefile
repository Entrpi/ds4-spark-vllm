CFLAGS ?= -O2 -Wall -Wextra -std=c11 -pthread
TARGET := ds4_dot_harness

$(TARGET): main.c ds4_dot.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

clean:
	rm -f $(TARGET)

.PHONY: clean
