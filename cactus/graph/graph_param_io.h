#pragma once

#include "graph.h"
#include <iosfwd>

namespace GraphParamIO {

void write_op_params(std::ostream& out, OpType op_type, const OpParams& params);
void read_op_params(std::istream& in, OpType op_type, OpParams& params);

} // namespace GraphParamIO
