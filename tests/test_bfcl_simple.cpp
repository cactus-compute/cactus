// Single-file BFCL simple_python function-calling benchmark.
//
// Builds as `test_bfcl_simple`; expects a Gemma-family or LFM2 model dir
// in CACTUS_TEST_MODEL. Runs 20 embedded BFCL cases through cactus_complete
// and reports a pass rate, matching BFCL's "simple" scoring (exactly one
// call, name matches one alternative, every ground-truth arg's value
// appears in its acceptable-values list; "" in the list means the arg may
// be omitted).

#include "../cactus/ffi/cactus_ffi.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {

struct Case {
    const char* id;
    const char* messages_json;
    const char* tools_json;
    const char* ground_truth_json;
};

// 20 BFCL simple_python cases (OpenAI-style schemas + BFCL acceptable-values ground truth).
static const Case kCases[] = {
    {"simple_python_0",
     R"R([{"role": "user", "content": "Find the area of a triangle with a base of 10 units and height of 5 units."}])R",
     R"R([{"type": "function", "function": {"name": "calculate_triangle_area", "description": "Calculate the area of a triangle given its base and height.", "parameters": {"type": "object", "properties": {"base": {"type": "integer", "description": "The base of the triangle."}, "height": {"type": "integer", "description": "The height of the triangle."}, "unit": {"type": "string", "description": "The unit of measure (defaults to 'units' if not specified)"}}, "required": ["base", "height"]}}}])R",
     R"R([{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}])R"},
    {"simple_python_1",
     R"R([{"role": "user", "content": "Calculate the factorial of 5 using math functions."}])R",
     R"R([{"type": "function", "function": {"name": "math.factorial", "description": "Calculate the factorial of a given number.", "parameters": {"type": "object", "properties": {"number": {"type": "integer", "description": "The number for which factorial needs to be calculated."}}, "required": ["number"]}}}])R",
     R"R([{"math.factorial": {"number": [5]}}])R"},
    {"simple_python_2",
     R"R([{"role": "user", "content": "Calculate the hypotenuse of a right triangle given the lengths of the other two sides as 4 and 5."}])R",
     R"R([{"type": "function", "function": {"name": "math.hypot", "description": "Calculate the Euclidean norm, sqrt(sum(squares)), the length of the vector from the origin to point (x, y) which is the hypotenuse of the right triangle.", "parameters": {"type": "object", "properties": {"x": {"type": "integer", "description": "The x-coordinate value."}, "y": {"type": "integer", "description": "The y-coordinate value."}, "z": {"type": "integer", "description": "Optional. The z-coordinate value. Default is 0."}}, "required": ["x", "y"]}}}])R",
     R"R([{"math.hypot": {"x": [4], "y": [5], "z": ["", 0]}}])R"},
    {"simple_python_3",
     R"R([{"role": "user", "content": "Find the roots of a quadratic equation with coefficients a=1, b=-3, c=2."}])R",
     R"R([{"type": "function", "function": {"name": "algebra.quadratic_roots", "description": "Find the roots of a quadratic equation ax^2 + bx + c = 0.", "parameters": {"type": "object", "properties": {"a": {"type": "integer", "description": "Coefficient of x^2."}, "b": {"type": "integer", "description": "Coefficient of x."}, "c": {"type": "integer", "description": "Constant term."}}, "required": ["a", "b", "c"]}}}])R",
     R"R([{"algebra.quadratic_roots": {"a": [1], "b": [-3], "c": [2]}}])R"},
    {"simple_python_4",
     R"R([{"role": "user", "content": "Solve a quadratic equation where a=2, b=6, and c=5"}])R",
     R"R([{"type": "function", "function": {"name": "solve_quadratic_equation", "description": "Function solves the quadratic equation and returns its roots.", "parameters": {"type": "object", "properties": {"a": {"type": "integer", "description": "Coefficient of x squared"}, "b": {"type": "integer", "description": "Coefficient of x"}, "c": {"type": "integer", "description": "Constant term in the quadratic equation."}}, "required": ["a", "b", "c"]}}}])R",
     R"R([{"solve_quadratic_equation": {"a": [2], "b": [6], "c": [5]}}])R"},
    {"simple_python_5",
     R"R([{"role": "user", "content": "Find all the roots of a quadratic equation given coefficients a = 3, b = -11, and c = -4."}])R",
     R"R([{"type": "function", "function": {"name": "solve_quadratic", "description": "Solve a quadratic equation given coefficients a, b, and c. If optional 'root_type' is 'real', the function will only return real roots. If not specified, function may return complex roots.", "parameters": {"type": "object", "properties": {"a": {"type": "integer", "description": "The coefficient of the squared term in the quadratic equation."}, "b": {"type": "integer", "description": "The coefficient of the linear term in the quadratic equation."}, "c": {"type": "integer", "description": "The constant term in the quadratic equation."}, "root_type": {"type": "string", "description": "The type of roots to return: 'real' for real roots, 'all' for both real and complex roots. Default value is 'real'."}}, "required": ["a", "b", "c"]}}}])R",
     R"R([{"solve_quadratic": {"a": [3], "b": [-11], "c": [-4], "root_type": ["all"]}}])R"},
    {"simple_python_6",
     R"R([{"role": "user", "content": "What are the roots of the quadratic equation where a=2, b=5 and c=3 ?"}])R",
     R"R([{"type": "function", "function": {"name": "solve_quadratic", "description": "Find the roots of a quadratic equation. Returns both roots.", "parameters": {"type": "object", "properties": {"a": {"type": "integer", "description": "Coefficient of x\u00b2."}, "b": {"type": "integer", "description": "Coefficient of x."}, "c": {"type": "integer", "description": "Constant term."}}, "required": ["a", "b", "c"]}}}])R",
     R"R([{"solve_quadratic": {"a": [2], "b": [5], "c": [3]}}])R"},
    {"simple_python_7",
     R"R([{"role": "user", "content": "What is the circumference of a circle with a radius of 4 inches?"}])R",
     R"R([{"type": "function", "function": {"name": "calculate_circumference", "description": "Calculates the circumference of a circle with a given radius.", "parameters": {"type": "object", "properties": {"radius": {"type": "integer", "description": "The radius of the circle in the unit given."}, "unit": {"type": "string", "description": "The unit of measurement for the radius. Default is 'cm'."}}, "required": ["radius"]}}}])R",
     R"R([{"calculate_circumference": {"radius": [4], "unit": ["inches", "in"]}}])R"},
    {"simple_python_8",
     R"R([{"role": "user", "content": "What's the area of a circle with a radius of 10?"}])R",
     R"R([{"type": "function", "function": {"name": "geometry.area_circle", "description": "Calculate the area of a circle given the radius.", "parameters": {"type": "object", "properties": {"radius": {"type": "integer", "description": "The radius of the circle."}, "units": {"type": "string", "description": "The units in which the radius is measured (defaults to 'meters')."}}, "required": ["radius"]}}}])R",
     R"R([{"geometry.area_circle": {"radius": [10], "units": ["meters", ""]}}])R"},
    {"simple_python_9",
     R"R([{"role": "user", "content": "Calculate the area of a circle with a radius of 5 units."}])R",
     R"R([{"type": "function", "function": {"name": "geometry.calculate_area_circle", "description": "Calculate the area of a circle given its radius.", "parameters": {"type": "object", "properties": {"radius": {"type": "integer", "description": "The radius of the circle."}, "unit": {"type": "string", "description": "The measurement unit of the radius (optional parameter, default is 'units')."}}, "required": ["radius"]}}}])R",
     R"R([{"geometry.calculate_area_circle": {"radius": [5], "unit": ["units", ""]}}])R"},
    {"simple_python_10",
     R"R([{"role": "user", "content": "Calculate the area of a right-angled triangle given the lengths of its base and height as 6cm and 10cm."}])R",
     R"R([{"type": "function", "function": {"name": "calculate_area", "description": "Calculate the area of a right-angled triangle given the lengths of its base and height.", "parameters": {"type": "object", "properties": {"base": {"type": "integer", "description": "The length of the base of the right-angled triangle."}, "height": {"type": "integer", "description": "The height of the right-angled triangle."}, "unit": {"type": "string", "description": "The unit of measure used. Defaults to 'cm'."}}, "required": ["base", "height"]}}}])R",
     R"R([{"calculate_area": {"base": [6], "height": [10], "unit": ["cm", ""]}}])R"},
    {"simple_python_11",
     R"R([{"role": "user", "content": "What is the area of a triangle with base of 10 units and height of 5 units?"}])R",
     R"R([{"type": "function", "function": {"name": "calculate_triangle_area", "description": "Calculate the area of a triangle using its base and height.", "parameters": {"type": "object", "properties": {"base": {"type": "integer", "description": "The length of the base of the triangle."}, "height": {"type": "integer", "description": "The height of the triangle from the base."}}, "required": ["base", "height"]}}}])R",
     R"R([{"calculate_triangle_area": {"base": [10], "height": [5]}}])R"},
    {"simple_python_12",
     R"R([{"role": "user", "content": "Calculate the circumference of a circle with radius 3"}])R",
     R"R([{"type": "function", "function": {"name": "geometry.circumference", "description": "Calculate the circumference of a circle given the radius.", "parameters": {"type": "object", "properties": {"radius": {"type": "integer", "description": "The radius of the circle."}, "units": {"type": "string", "description": "Units for the output circumference measurement. Default is 'cm'."}}, "required": ["radius"]}}}])R",
     R"R([{"geometry.circumference": {"radius": [3], "units": ["cm", ""]}}])R"},
    {"simple_python_13",
     R"R([{"role": "user", "content": "Calculate the area under the curve y=x^2 from x=1 to x=3."}])R",
     R"R([{"type": "function", "function": {"name": "calculate_area_under_curve", "description": "Calculate the area under a mathematical function within a given interval.", "parameters": {"type": "object", "properties": {"function": {"type": "string", "description": "The mathematical function as a string."}, "interval": {"type": "array", "items": {"type": "number"}, "description": "An array that defines the interval to calculate the area under the curve from the start to the end point."}, "method": {"type": "string", "description": "The numerical method to approximate the area under the curve. The default value is 'trapezoidal'."}}, "required": ["function", "interval"]}}}])R",
     R"R([{"calculate_area_under_curve": {"function": ["x**2", "lambda x: x**2", "y=x**2"], "interval": [[1.0, 3.0]], "method": ["", "trapezoidal"]}}])R"},
    {"simple_python_14",
     R"R([{"role": "user", "content": "Calculate the derivative of the function 3x^2 + 2x - 1."}])R",
     R"R([{"type": "function", "function": {"name": "calculate_derivative", "description": "Calculate the derivative of a polynomial function.", "parameters": {"type": "object", "properties": {"function": {"type": "string", "description": "The polynomial function."}, "x_value": {"type": "number", "description": "The x-value at which the derivative is calculated. Optional, default to 0.00."}}, "required": ["function"]}}}])R",
     R"R([{"calculate_derivative": {"function": ["3x**2 + 2x - 1", "lambda x: 3x**2 + 2x - 1"], "x_value": ["", 0.0]}}])R"},
    {"simple_python_15",
     R"R([{"role": "user", "content": "Calculate the area under the curve from x = -2 to x = 3 for the function y = x^3 using simpson method."}])R",
     R"R([{"type": "function", "function": {"name": "integrate", "description": "Calculate the area under a curve for a specified function between two x values.", "parameters": {"type": "object", "properties": {"function": {"type": "string", "description": "The function to integrate, represented as a string. For example, 'x^3'"}, "start_x": {"type": "integer", "description": "The starting x-value to integrate over."}, "end_x": {"type": "integer", "description": "The ending x-value to integrate over."}, "method": {"type": "string", "description": "The method of numerical integration to use. Choices are 'trapezoid' or 'simpson'. Default is 'trapezoid'."}}, "required": ["function", "start_x", "end_x"]}}}])R",
     R"R([{"integrate": {"function": ["x**3", "lambda x: x**3"], "start_x": [-2], "end_x": [3], "method": ["simpson"]}}])R"},
    {"simple_python_16",
     R"R([{"role": "user", "content": "Calculate the derivative of the function 2x^2 at x = 1."}])R",
     R"R([{"type": "function", "function": {"name": "calculus.derivative", "description": "Compute the derivative of a function at a specific value.", "parameters": {"type": "object", "properties": {"function": {"type": "string", "description": "The function to calculate the derivative of."}, "value": {"type": "integer", "description": "The value where the derivative needs to be calculated at."}, "function_variable": {"type": "string", "description": "The variable present in the function, for instance x or y, etc. Default is 'x'."}}, "required": ["function", "value"]}}}])R",
     R"R([{"calculus.derivative": {"function": ["2x**2", "lambda x: 2x**2"], "value": [1], "function_variable": ["x", ""]}}])R"},
    {"simple_python_17",
     R"R([{"role": "user", "content": "Find the prime factors of 450"}])R",
     R"R([{"type": "function", "function": {"name": "get_prime_factors", "description": "Function to retrieve prime factors of a number.", "parameters": {"type": "object", "properties": {"number": {"type": "integer", "description": "Number for which prime factors need to be calculated"}, "formatted": {"type": "boolean", "description": "Return formatted string if true, array if false. Default is true."}}, "required": ["number", "formatted"]}}}])R",
     R"R([{"get_prime_factors": {"number": [450], "formatted": [true, ""]}}])R"},
    {"simple_python_18",
     R"R([{"role": "user", "content": "Find the prime factors of the number 123456."}])R",
     R"R([{"type": "function", "function": {"name": "number_analysis.prime_factors", "description": "Compute the prime factors of a number.", "parameters": {"type": "object", "properties": {"number": {"type": "integer", "description": "The number to be factored."}}, "required": ["number"]}}}])R",
     R"R([{"number_analysis.prime_factors": {"number": [123456]}}])R"},
    {"simple_python_19",
     R"R([{"role": "user", "content": "Calculate the greatest common divisor of two numbers: 40 and 50"}])R",
     R"R([{"type": "function", "function": {"name": "math.gcd", "description": "Compute the greatest common divisor of two numbers", "parameters": {"type": "object", "properties": {"num1": {"type": "integer", "description": "The first number."}, "num2": {"type": "integer", "description": "The second number."}}, "required": ["num1", "num2"]}}}])R",
     R"R([{"math.gcd": {"num1": [40], "num2": [50]}}])R"},
};

// ---------- minimal JSON cursor helpers ----------

void skip_ws(const std::string& s, size_t& p) {
    while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) p++;
}

size_t end_of_string(const std::string& s, size_t p) {
    size_t q = p + 1;
    while (q < s.size()) {
        if (s[q] == '\\' && q + 1 < s.size()) { q += 2; continue; }
        if (s[q] == '"') return q + 1;
        q++;
    }
    return s.size();
}

size_t end_of_bracketed(const std::string& s, size_t p, char open, char close) {
    int depth = 0;
    bool in_str = false;
    while (p < s.size()) {
        char c = s[p];
        if (in_str) {
            if (c == '\\' && p + 1 < s.size()) { p += 2; continue; }
            if (c == '"') in_str = false;
        } else {
            if (c == '"') in_str = true;
            else if (c == open) depth++;
            else if (c == close) { depth--; if (depth == 0) return p + 1; }
        }
        p++;
    }
    return s.size();
}

size_t end_of_value(const std::string& s, size_t p) {
    skip_ws(s, p);
    if (p >= s.size()) return p;
    char c = s[p];
    if (c == '"') return end_of_string(s, p);
    if (c == '{') return end_of_bracketed(s, p, '{', '}');
    if (c == '[') return end_of_bracketed(s, p, '[', ']');
    size_t q = p;
    while (q < s.size() && s[q] != ',' && s[q] != ']' && s[q] != '}' &&
           !std::isspace(static_cast<unsigned char>(s[q]))) q++;
    return q;
}

std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) a++;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b-1]))) b--;
    return s.substr(a, b - a);
}

std::string extract_field(const std::string& obj, const std::string& key) {
    size_t p = 0;
    skip_ws(obj, p);
    if (p >= obj.size() || obj[p] != '{') return {};
    p++;
    while (p < obj.size()) {
        skip_ws(obj, p);
        if (p >= obj.size() || obj[p] == '}') break;
        if (obj[p] == ',') { p++; continue; }
        if (obj[p] != '"') break;
        size_t key_end = end_of_string(obj, p);
        std::string k = obj.substr(p + 1, key_end - p - 2);
        p = key_end;
        skip_ws(obj, p);
        if (p < obj.size() && obj[p] == ':') p++;
        skip_ws(obj, p);
        size_t val_end = end_of_value(obj, p);
        if (k == key) return obj.substr(p, val_end - p);
        p = val_end;
    }
    return {};
}

std::vector<std::string> array_items(const std::string& arr) {
    std::vector<std::string> out;
    size_t p = 0;
    skip_ws(arr, p);
    if (p >= arr.size() || arr[p] != '[') return out;
    p++;
    while (p < arr.size()) {
        skip_ws(arr, p);
        if (p >= arr.size() || arr[p] == ']') break;
        if (arr[p] == ',') { p++; continue; }
        size_t end = end_of_value(arr, p);
        out.push_back(arr.substr(p, end - p));
        p = end;
    }
    return out;
}

std::map<std::string, std::string> object_fields(const std::string& obj) {
    std::map<std::string, std::string> out;
    size_t p = 0;
    skip_ws(obj, p);
    if (p >= obj.size() || obj[p] != '{') return out;
    p++;
    while (p < obj.size()) {
        skip_ws(obj, p);
        if (p >= obj.size() || obj[p] == '}') break;
        if (obj[p] == ',') { p++; continue; }
        if (obj[p] != '"') break;
        size_t key_end = end_of_string(obj, p);
        std::string k = obj.substr(p + 1, key_end - p - 2);
        p = key_end;
        skip_ws(obj, p);
        if (p < obj.size() && obj[p] == ':') p++;
        skip_ws(obj, p);
        size_t val_end = end_of_value(obj, p);
        out[k] = obj.substr(p, val_end - p);
        p = val_end;
    }
    return out;
}

std::string decode_string_literal(const std::string& quoted) {
    if (quoted.size() < 2 || quoted.front() != '"' || quoted.back() != '"') return quoted;
    std::string out;
    for (size_t i = 1; i + 1 < quoted.size(); i++) {
        char c = quoted[i];
        if (c == '\\' && i + 2 < quoted.size()) {
            char n = quoted[i + 1];
            if (n == 'n') out += '\n';
            else if (n == 't') out += '\t';
            else if (n == '"') out += '"';
            else if (n == '\\') out += '\\';
            else out += n;
            i++;
        } else {
            out += c;
        }
    }
    return out;
}

// ---------- value matcher ----------

// Gemma 4 sometimes emits bare identifiers instead of quoted strings:
// {"word": TANGERINE} vs {"word": "TANGERINE"}. Treat them the same.
std::string maybe_quote_bare_identifier(const std::string& s) {
    if (s.empty() || s.front() == '"' || s.front() == '{' || s.front() == '[') return s;
    for (char c : s) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.')) return s;
    }
    if (std::isdigit(static_cast<unsigned char>(s.front()))) return s;
    return "\"" + s + "\"";
}

// Accept x^n and x**n as equivalent, plus whitespace-insensitive.
std::string normalize_math(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); i++) {
        char c = s[i];
        if (std::isspace(static_cast<unsigned char>(c))) continue;
        if (c == '^') { out += "**"; continue; }
        out += c;
    }
    return out;
}

bool values_equal(const std::string& a_raw, const std::string& b_raw) {
    std::string a = maybe_quote_bare_identifier(trim(a_raw));
    std::string b = maybe_quote_bare_identifier(trim(b_raw));
    if (a == b) return true;
    try {
        size_t ea = 0, eb = 0;
        double da = std::stod(a, &ea);
        double db = std::stod(b, &eb);
        if (ea == a.size() && eb == b.size() && std::fabs(da - db) < 1e-9) return true;
    } catch (...) {}
    if (!a.empty() && !b.empty() && a.front() == '"' && b.front() == '"') {
        std::string da = decode_string_literal(a);
        std::string db = decode_string_literal(b);
        return da == db || normalize_math(da) == normalize_math(db);
    }
    if (!a.empty() && !b.empty() && a.front() == '[' && b.front() == '[') {
        auto ai = array_items(a);
        auto bi = array_items(b);
        if (ai.size() != bi.size()) return false;
        for (size_t i = 0; i < ai.size(); i++) if (!values_equal(ai[i], bi[i])) return false;
        return true;
    }
    return false;
}

bool args_match(const std::string& call_args_json, const std::string& gt_args_json) {
    auto call_args = object_fields(call_args_json);
    auto gt_fields = object_fields(gt_args_json);
    for (const auto& kv : gt_fields) {
        const std::string& key = kv.first;
        auto acceptable = array_items(kv.second);
        bool omission_ok = false;
        for (const auto& a : acceptable) if (trim(a) == "\"\"") { omission_ok = true; break; }

        auto it = call_args.find(key);
        if (it == call_args.end()) {
            if (omission_ok) continue;
            return false;
        }
        bool any = false;
        for (const auto& a : acceptable) {
            std::string av = trim(a);
            if (av == "\"\"") continue;
            if (values_equal(it->second, a)) { any = true; break; }
        }
        if (!any) return false;
    }
    return true;
}

// ---------- runner ----------

bool run_case(cactus_model_t model, const Case& c, std::string& note, std::string& got) {
    const char* options = R"({
        "max_tokens": 512,
        "temperature": 0,
        "top_k": 1,
        "force_tools": true,
        "enable_thinking_if_supported": false,
        "auto_handoff": false,
        "telemetry_enabled": false
    })";

    cactus_reset(model);

    std::vector<char> resp(32 * 1024);
    int rc = cactus_complete(model, c.messages_json, resp.data(), resp.size(),
                             options, c.tools_json, nullptr, nullptr, nullptr, 0);
    if (rc < 0) {
        note = "cactus_complete failed (" + std::to_string(rc) + ")";
        return false;
    }
    std::string response(resp.data());

    auto fcs = array_items(extract_field(response, "function_calls"));
    if (fcs.empty()) {
        note = "no function_calls produced";
        return false;
    }
    if (fcs.size() > 1) {
        note = "expected 1 call, got " + std::to_string(fcs.size());
        return false;
    }
    got = fcs[0];

    std::string name_raw = extract_field(fcs[0], "name");
    std::string args_raw = extract_field(fcs[0], "arguments");
    if (name_raw.empty() || args_raw.empty()) { note = "malformed call"; return false; }
    std::string model_name = decode_string_literal(name_raw);

    for (const auto& alt : array_items(trim(c.ground_truth_json))) {
        auto entries = object_fields(alt);
        if (entries.size() != 1) continue;
        const auto& gt_name = entries.begin()->first;
        const auto& gt_args = entries.begin()->second;
        if (model_name != gt_name) { note = "name mismatch: got '" + model_name + "' expected '" + gt_name + "'"; continue; }
        if (args_match(args_raw, gt_args)) return true;
        note = "args don't match " + gt_name;
    }
    if (note.empty()) note = "no ground-truth alternative matched";
    return false;
}

} // namespace

int main() {
    const char* model_path = std::getenv("CACTUS_TEST_MODEL");
    if (!model_path) {
        std::fprintf(stderr, "SKIP: set CACTUS_TEST_MODEL to a tool-capable model directory\n");
        return 0;
    }

    std::printf("BFCL simple_python (%zu cases)\nModel: %s\n%s\n",
                sizeof(kCases) / sizeof(kCases[0]), model_path, std::string(60, '-').c_str());

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) {
        const char* err = cactus_get_last_error();
        std::fprintf(stderr, "Failed to init model (%s)\n", err ? err : "");
        return 1;
    }

    size_t passed = 0;
    const size_t case_count = sizeof(kCases) / sizeof(kCases[0]);
    for (size_t i = 0; i < case_count; i++) {
        const auto& c = kCases[i];
        std::string note, got;
        bool ok = run_case(model, c, note, got);
        if (ok) passed++;
        std::printf("%s %-24s", ok ? "[PASS]" : "[FAIL]", c.id);
        if (!ok) {
            std::printf("  -- %s", note.c_str());
            if (!got.empty()) std::printf("\n         got: %s", got.c_str());
        }
        std::printf("\n");
    }

    cactus_destroy(model);

    std::printf("%s\nPass rate: %zu/%zu (%.1f%%)\n",
                std::string(60, '-').c_str(), passed, case_count,
                100.0 * static_cast<double>(passed) / static_cast<double>(case_count));
    return passed == case_count ? 0 : 1;
}
