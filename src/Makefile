CC ?= gcc
AR ?= ar
CFLAGS ?= -O2 -fPIC -I.
LDFLAGS ?=
LIBS ?= -llapack -lblas -lm

CORE_OBJS := brlmm.o brlmm_utils.o
TEST_NATIVE := test_native

all: libbrlmm_core.a libbrlmm_core.so $(TEST_NATIVE)

libbrlmm_core.a: $(CORE_OBJS)
	$(AR) rcs $@ $(CORE_OBJS)

libbrlmm_core.so: $(CORE_OBJS)
	$(CC) -shared -o $@ $(CORE_OBJS) $(LDFLAGS) $(LIBS)

$(TEST_NATIVE): $(CORE_OBJS) ../tests/test_native.o
	$(CC) -o $@ $(CORE_OBJS) ../tests/test_native.o $(LDFLAGS) $(LIBS)

%.o: %.c brlmm.h brlmm_utils.h
	$(CC) $(CFLAGS) -c $< -o $@

../tests/test_native.o: ../tests/test_native.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

clean:
	rm -f $(CORE_OBJS) libbrlmm_core.a libbrlmm_core.so ../tests/test_native.o $(TEST_NATIVE)

.PHONY: all clean
