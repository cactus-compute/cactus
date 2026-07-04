file(READ "${IN}" SRC_HEX HEX)
string(REGEX REPLACE "(..)" "0x\\1," SRC_BYTES "${SRC_HEX}")
file(WRITE "${OUT}"
  "static const char kCactusMSL[] = {${SRC_BYTES}0x00};\n")
