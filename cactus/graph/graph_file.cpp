#include "graph.h"
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

namespace GraphFile {
    
    void save_node(CactusGraph& graph, size_t node_id, const std::string& filename) {
        graph.execute();
        void* data = graph.get_output(node_id);
        
        const auto& buffer = graph.get_output_buffer(node_id);
        const auto& shape = buffer.shape;
        Precision precision = buffer.precision;
        
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }
        
        uint32_t ndim = static_cast<uint32_t>(shape.size());
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        
        for (size_t dim : shape) {
            uint64_t dim_val = static_cast<uint64_t>(dim);
            file.write(reinterpret_cast<const char*>(&dim_val), sizeof(dim_val));
        }
        
        uint32_t prec_val = static_cast<uint32_t>(precision);
        file.write(reinterpret_cast<const char*>(&prec_val), sizeof(prec_val));
        
        size_t total_elements = 1;
        for (size_t dim : shape) {
            total_elements *= dim;
        }
        
        size_t element_size = PrecisionTraits::size_of(precision);
        size_t byte_size = total_elements * element_size;
        uint64_t size_val = static_cast<uint64_t>(byte_size);
        file.write(reinterpret_cast<const char*>(&size_val), sizeof(size_val));
        
        if (precision == Precision::INT8) {
            float quantization_scale = buffer.quantization_scale;
            file.write(reinterpret_cast<const char*>(&quantization_scale), sizeof(quantization_scale));
        }
        
        file.write(static_cast<const char*>(data), byte_size);
        
        if (!file) {
            throw std::runtime_error("Error writing node data to file: " + filename);
        }
    }
    
    LoadedNode load_into_graph(CactusGraph& graph, const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for reading: " + filename);
        }
        
        uint32_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        
        std::vector<size_t> shape(ndim);
        for (uint32_t i = 0; i < ndim; i++) {
            uint64_t dim_val;
            file.read(reinterpret_cast<char*>(&dim_val), sizeof(dim_val));
            shape[i] = static_cast<size_t>(dim_val);
        }
        
        uint32_t prec_val;
        file.read(reinterpret_cast<char*>(&prec_val), sizeof(prec_val));
        Precision precision = static_cast<Precision>(prec_val);
        
        uint64_t size_val;
        file.read(reinterpret_cast<char*>(&size_val), sizeof(size_val));
        size_t byte_size = static_cast<size_t>(size_val);
        
        float quantization_scale = 1.0f;
        if (precision == Precision::INT8) {
            file.read(reinterpret_cast<char*>(&quantization_scale), sizeof(quantization_scale));
        }
        
        if (!file) {
            throw std::runtime_error("Error reading file header: " + filename);
        }
        
        std::vector<char> buffer(byte_size);
        file.read(buffer.data(), byte_size);
        
        if (!file || file.gcount() != static_cast<std::streamsize>(byte_size)) {
            throw std::runtime_error("Error reading node data: " + filename);
        }
        
        size_t node_id = graph.input(shape, precision);
        if (precision == Precision::INT8) {
            graph.set_quantization_scale(node_id, quantization_scale);
        }
        graph.set_input(node_id, buffer.data(), precision);
        
        return {node_id, shape, precision, byte_size};
    }
    
    MappedFile mmap_load(const std::string& filename);
}

GraphFile::MappedFile::MappedFile(const std::string& filename) 
    : fd_(-1), mapped_data_(nullptr), file_size_(0), data_offset_(0) {
    fd_ = open(filename.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("Cannot open file for mapping: " + filename);
    }
    
    struct stat st;
    if (fstat(fd_, &st) == -1) {
        close(fd_);
        throw std::runtime_error("Cannot get file size: " + filename);
    }
    file_size_ = st.st_size;
    
    mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_data_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("Cannot map file: " + filename);
    }
    
    parse_header();
}

GraphFile::MappedFile::~MappedFile() {
    if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
        madvise(mapped_data_, file_size_, MADV_DONTNEED);
        munmap(mapped_data_, file_size_);
        mapped_data_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}

GraphFile::MappedFile::MappedFile(MappedFile&& other) noexcept 
    : fd_(other.fd_), mapped_data_(other.mapped_data_), file_size_(other.file_size_),
      data_offset_(other.data_offset_), shape_(std::move(other.shape_)),
      precision_(other.precision_), byte_size_(other.byte_size_), quantization_scale_(other.quantization_scale_) {
    other.fd_ = -1;
    other.mapped_data_ = nullptr;
    other.file_size_ = 0;
}

GraphFile::MappedFile& GraphFile::MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
            munmap(mapped_data_, file_size_);
        }
        if (fd_ != -1) {
            close(fd_);
        }
        
        fd_ = other.fd_;
        mapped_data_ = other.mapped_data_;
        file_size_ = other.file_size_;
        data_offset_ = other.data_offset_;
        shape_ = std::move(other.shape_);
        precision_ = other.precision_;
        byte_size_ = other.byte_size_;
        quantization_scale_ = other.quantization_scale_;
        
        other.fd_ = -1;
        other.mapped_data_ = nullptr;
        other.file_size_ = 0;
    }
    return *this;
}

const std::vector<size_t>& GraphFile::MappedFile::shape() const { 
    return shape_; 
}

Precision GraphFile::MappedFile::precision() const { 
    return precision_; 
}

size_t GraphFile::MappedFile::byte_size() const { 
    return byte_size_; 
}

float GraphFile::MappedFile::quantization_scale() const {
    return quantization_scale_;
}


void* GraphFile::MappedFile::data() {
    return static_cast<char*>(mapped_data_) + data_offset_;
}

const void* GraphFile::MappedFile::data() const {
    return static_cast<const char*>(mapped_data_) + data_offset_;
}

template<typename T>
const T* GraphFile::MappedFile::typed_data() const {
    return static_cast<const T*>(data());
}

GraphFile::LoadedNode GraphFile::MappedFile::load_into_graph(CactusGraph& graph) const {
    size_t node_id = graph.input(shape_, precision_);
    graph.set_external_input(node_id, const_cast<void*>(data()), precision_);
    if (precision_ == Precision::INT8) {
        graph.set_quantization_scale(node_id, quantization_scale_);
    }
    return {node_id, shape_, precision_, byte_size_};
}

void GraphFile::MappedFile::parse_header() {
    const char* ptr = static_cast<const char*>(mapped_data_);
    size_t offset = 0;
    
    const size_t min_header_size = sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t);
    if (file_size_ < min_header_size) {
        throw std::runtime_error("File too small: insufficient data for header");
    }
    
    uint32_t ndim = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);
    
   const size_t dims_size = ndim * sizeof(uint64_t);
    if (offset + dims_size + sizeof(uint32_t) + sizeof(uint64_t) > file_size_) {
        throw std::runtime_error("File corrupted: insufficient data for header with " + std::to_string(ndim) + " dimensions");
    }
    
    shape_.resize(ndim);
    for (uint32_t i = 0; i < ndim; i++) {
        uint64_t dim_val = *reinterpret_cast<const uint64_t*>(ptr + offset);
        shape_[i] = static_cast<size_t>(dim_val);
        offset += sizeof(uint64_t);
    }
    
    uint32_t prec_val = *reinterpret_cast<const uint32_t*>(ptr + offset);
    precision_ = static_cast<Precision>(prec_val);
    offset += sizeof(uint32_t);
    
    uint64_t size_val = *reinterpret_cast<const uint64_t*>(ptr + offset);
    byte_size_ = static_cast<size_t>(size_val);
    offset += sizeof(uint64_t);
    
    if (precision_ == Precision::INT8) {
        if (offset + sizeof(float) > file_size_) {
            throw std::runtime_error("File corrupted: missing quantization parameters for INT8 tensor");
        }
        quantization_scale_ = *reinterpret_cast<const float*>(ptr + offset);
        offset += sizeof(float);
    } else {
        quantization_scale_ = 1.0f;
    }
    
    data_offset_ = offset;
    
    if (data_offset_ + byte_size_ > file_size_) {
        throw std::runtime_error("File corrupted: data extends beyond file size");
    }
}

template const int8_t* GraphFile::MappedFile::typed_data<int8_t>() const;
template const float* GraphFile::MappedFile::typed_data<float>() const;
template const uint16_t* GraphFile::MappedFile::typed_data<uint16_t>() const;
template const uint8_t* GraphFile::MappedFile::typed_data<uint8_t>() const;

GraphFile::MappedFile GraphFile::mmap_load(const std::string& filename) {
    return MappedFile(filename);
} 