//
// https://github.com/SteveKChiu/lua-intf
//
// TensorView: Zero-copy array wrapper for efficient Lua-C++ data exchange
//
// Copyright 2026, model_infer contributors
//
// The MIT License (http://www.opensource.org/licenses/mit-license.php)
//

#ifndef TENSORVIEW_H
#define TENSORVIEW_H

#include <memory>
#include <stdexcept>
#include <cstddef>

/**
 * TensorView provides zero-copy access to C++ array data from Lua.
 * 
 * Instead of copying large tensors when passing between C++ and Lua,
 * TensorView wraps a raw pointer and provides Lua-side element access
 * via get/set methods that support Lua 1-based indexing.
 * 
 * Lifetime management: Uses shared_ptr<void> to keep underlying data alive.
 * The owner shared_ptr must outlive or be captured by the view.
 * 
 * Performance: Eliminates memory copies for large arrays (e.g., 640×640×3 images).
 * A 10MB tensor passed 1000 times uses ~10MB instead of ~10GB with copying.
 * 
 * Usage:
 * @code
 *   // C++ side: create view wrapping existing data
 *   auto data = std::make_shared<std::vector<float>>(1000000, 0.0f);
 *   TensorView<float> view(data->data(), data->size(), data);
 * 
 *   // Bind to Lua
 *   LuaBinding(L).beginClass<TensorView<float>>("FloatTensorView")
 *       .addConstructor(LUA_ARGS())
 *       .addFunction("get", &TensorView<float>::get)
 *       .addFunction("set", &TensorView<float>::set)
 *       .addFunction("__len", &TensorView<float>::length)
 *   .endClass();
 * 
 *   // Lua side: access without copying
 *   local view = createView()
 *   print(#view)           -- length
 *   print(view:get(1))     -- access element
 *   view:set(1, 3.14)      -- modify element
 * @endcode
 */
template<typename T>
class TensorView {
private:
    T* data_;
    size_t length_;
    std::shared_ptr<void> owner_;  // Keeps data alive
    
public:
    /**
     * Default constructor - creates empty view.
     */
    TensorView() : data_(nullptr), length_(0), owner_(nullptr) {}
    
    /**
     * Create view from raw pointer with explicit ownership.
     * 
     * @param data Raw pointer to array data (must remain valid)
     * @param len Number of elements in array
     * @param owner Shared pointer keeping data alive (optional but recommended)
     * 
     * Note: If owner is null, caller must ensure data remains valid for view lifetime.
     */
    TensorView(T* data, size_t len, std::shared_ptr<void> owner = nullptr)
        : data_(data), length_(len), owner_(owner) {}
    
    /**
     * Get element at 1-based Lua index (converted to 0-based C++ internally).
     * 
     * @param idx 1-based index (Lua convention)
     * @return Element value
     * @throws std::out_of_range if index is invalid
     */
    T get(int idx) const {
        if (idx < 1 || idx > static_cast<int>(length_)) {
            throw std::out_of_range("TensorView: index out of range");
        }
        return data_[idx - 1];  // Lua 1-based → C++ 0-based
    }
    
    /**
     * Set element at 1-based Lua index (converted to 0-based C++ internally).
     * 
     * @param idx 1-based index (Lua convention)
     * @param val New value to set
     * @throws std::out_of_range if index is invalid
     */
    void set(int idx, T val) {
        if (idx < 1 || idx > static_cast<int>(length_)) {
            throw std::out_of_range("TensorView: index out of range");
        }
        data_[idx - 1] = val;
    }
    
    /**
     * Get view length (for __len metamethod or direct access).
     * Returns int for Lua compatibility (size_t causes template errors)
     * 
     * @return Number of elements in view
     */
    int length() const { return static_cast<int>(length_); }
    
    /**
     * Get raw size_t length
     */
    size_t size() const { return length_; }
    
    /**
     * Get raw pointer (for C++ interop and advanced use cases).
     * 
     * @return Raw pointer to underlying data
     * 
     * Warning: Ensure data remains valid when using this pointer.
     */
    T* data() const { return data_; }
    
    /**
     * Check if view is empty.
     * 
     * @return true if view has no elements
     */
    bool empty() const { return length_ == 0; }
    
    /**
     * Check if view is valid (has data pointer).
     * 
     * @return true if view points to valid data
     */
    bool isValid() const { return data_ != nullptr; }
};

#endif  // TENSORVIEW_H
