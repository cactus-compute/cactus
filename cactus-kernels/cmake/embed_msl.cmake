file(READ "${IN}" SRC)
if(SRC MATCHES "\\)${DELIM}\"")
  message(FATAL_ERROR "embed_msl: source contains the raw-string delimiter )${DELIM}\" - choose another DELIM")
endif()
file(WRITE "${OUT}"
  "static const char* kCactusMSL = R\"${DELIM}(\n"
  "${SRC}\n"
  ")${DELIM}\";\n")
