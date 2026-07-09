if(NOT DEFINED SYM)
  set(SYM kCactusMSL)
endif()
file(READ "${IN}" SRC_HEX HEX)
string(REGEX REPLACE "(..)" "0x\\1," SRC_BYTES "${SRC_HEX}")
file(WRITE "${OUT}"
  "static const char ${SYM}[] = {${SRC_BYTES}0x00};\n")
