// Mesh.hpp 
//
// list of fornfdm::Vector representing
// a disretization of space in any # of dimensions 
// 
// JAF 4/13/2026 

#ifndef FORNFDM_MESH_H
#define FORNFDM_MESH_H 

#include<array> // std::array
#include<cassert>
#include<vector> // std::vector
#include<new>
#include<memory> // std::shared_ptr
#include<memory_resource>
#include<Eigen/Core> 

#include"types.hpp"

namespace fornfdm {

class Mesh : public std::enable_shared_from_this<Mesh>
{
  private:
    // Type Defs -----------------------------
    struct hidden_key{};
    static constexpr std::size_t kSboCutoff = 3;
    static constexpr std::uint8_t kIsOnHeapBit = (1u << 7);
    struct HeapStorage
    {
      Axis* pointer;
      std::size_t size;
    };
    union Storage
    {
      alignas(Axis) unsigned char bytes[sizeof(Axis)*kSboCutoff];
      HeapStorage heap_storage;
    };

  public:
    // Constructors + Destructor --------------- 

    // from number of axes. stops shared_ptr from initializing to nullptr. 
    Mesh(std::size_t ndims, std::pmr::memory_resource* mr, hidden_key key)
    {
      if(ndims == 0) throw std::invalid_argument("must pass ndims > 0 to Mesh::create");
      const bool on_heap = ndims > kSboCutoff;
      Axis* axes_ptr = on_heap ?
                        reinterpret_cast<Axis*>(mr->allocate(ndims*sizeof(Axis), alignof(Axis))) :
                        reinterpret_cast<Axis*>(m_storage.bytes); 

      std::size_t built = 0;
      try
      {
        for(; built<ndims; ++built)
        {
          new (static_cast<void*>(axes_ptr+built)) Axis(mr);
        }
      }
      catch(...)
      {
        for(std::size_t i=0; i<built; ++i)
        {
          axes_ptr[i].~Axis();
        }
        if(on_heap) mr->deallocate(axes_ptr, ndims*sizeof(Axis), alignof(Axis));
        throw;
      }
      if(on_heap)
      {
        m_storage.heap_storage = HeapStorage{axes_ptr, ndims};
        m_metadata = kIsOnHeapBit;
      }
      else
      {
        m_metadata = ndims;
      }
    }

    // static factory method. enforce shared_ptr only
    static std::shared_ptr<Mesh> create(std::size_t ndims, std::pmr::memory_resource* mr = std::pmr::get_default_resource())
    {
      return std::make_shared<Mesh>(ndims,mr,hidden_key{});
    }

    // Copy 
    Mesh(const Mesh& other)=delete; 
    Mesh& operator=(const Mesh& other)=delete;
    
    // destructors
    ~Mesh()
    {
      const bool on_heap = ((m_metadata & kIsOnHeapBit)!=0);

      Axis* axes_ptr = on_heap ? 
              m_storage.heap_storage.pointer :
              std::launder(reinterpret_cast<Axis*>(m_storage.bytes));

      std::pmr::memory_resource* mr = axes_ptr[0].get_allocator().resource();

      std::size_t end = numDims();
      for(std::size_t i=0; i<end; ++i)
      {
        axes_ptr[i].~Axis();
      }

      if(on_heap) mr->deallocate(axes_ptr, end*sizeof(Axis),alignof(Axis));
    }

    // Member Functions --------------

    std::shared_ptr<Mesh> clone()
    {
      const Axis* axes_ptr = get_axes_ptr();
      std::size_t ndims = numDims();
      auto result = create(ndims, axes_ptr[0].get_allocator().resource());
      auto* dest_ptr = result->get_axes_ptr();
      for(std::size_t i=0; i<ndims; ++i)
      {
        dest_ptr[i] = axes_ptr[i];
      }
      return result;
    }
    
    // number of dimensions 
    std::size_t numDims() const 
    {
      const bool on_heap = (m_metadata & kIsOnHeapBit)!=0;
      return on_heap ? 
          m_storage.heap_storage.size :
          m_metadata; 
    }

    // size of a specific axis 
    std::size_t sizeOfDim(std::size_t i) const {assert(i<numDims()); return get_axes_ptr()[i].size();} 

    // get a specific axis
    auto& getAxis(std::size_t i){ assert(i<numDims()); return get_axes_ptr()[i]; }
    const auto& getAxis(std::size_t i) const { assert(i<numDims()); return get_axes_ptr()[i]; }

    // set a specific axis from an eigen expression
    template<typename Derived>
    SharedMesh setAxis(std::size_t i, const Eigen::MatrixBase<Derived>& xpr)
    {
      assert(i < numDims());
      std::size_t sz = xpr.size();
      Axis* ptr = get_axes_ptr();
      std::pmr::memory_resource* mr = ptr[0].get_allocator().resource(); 
      Axis temp{mr};
      temp.resize(sz);
      temp.swap(ptr[i]); // no deep copies.
      Eigen::Map<fornfdm::Vector> view(ptr[i].data(), sz);
      view = xpr;
      return shared_from_this();
    }

    // full size of XD mesh. i.e. axis1.size() * ... * axisn.size()
    std::size_t sizesProduct() const 
    {
      std::size_t p = 1; 
      const auto* axes = get_axes_ptr();
      std::size_t end = numDims();
      for(auto idx=0; idx<end; ++idx){
        p *= axes[idx].size(); 
      }
      return p; 
    } 

    // product of axes up to dim exclusively [first, dim)
    std::size_t sizesMiddleProduct(std::size_t start, std::size_t end) const 
    {
      assert((start <= end) && "start index must be <= end index for middle product"); 
      assert((end <= numDims()) && "end index must be <= # of numDims in MeshXD"); 
      std::size_t prod = 1; 
      const auto* ptr = get_axes_ptr();
      for(auto idx=start; idx<end; ++idx){
        prod *= ptr[idx].size(); 
      }
      return prod; 
    } 

  private:
    Axis* get_axes_ptr()
    {
      bool on_heap = ((m_metadata & kIsOnHeapBit)!=0);
      return on_heap ? 
            m_storage.heap_storage.pointer :
            std::launder(reinterpret_cast<Axis*>(m_storage.bytes));
    }

    const Axis* get_axes_ptr() const
    {
      bool on_heap = ((m_metadata & kIsOnHeapBit)!=0);
      return on_heap ? 
            m_storage.heap_storage.pointer :
            std::launder(reinterpret_cast<const Axis*>(m_storage.bytes));
    }

    // member data ----------------------------
    Storage m_storage;
    std::uint8_t m_metadata;

};

template<typename... Args> 
auto make_Mesh(Args... args)
{
  return std::make_shared<Mesh>(args...); 
}

} // end namespace fornfdm 

#include "Coordinate.hpp" 

namespace fornfdm{ 
auto discretize(const fornfdm::Mesh* m, fornfdm::Scalar a)
{
  std::size_t s = m->sizesProduct(); 
  auto lam = [a](std::size_t i){ return a; }; 
  return Eigen::CwiseNullaryOp<decltype(lam), fornfdm::Vector>(s, 1, lam); 
}

template<class Callable>
auto discretize(const fornfdm::Mesh* m, const Callable& func)
{
  std::size_t s = m->sizesProduct(); 
  constexpr std::size_t N = fornfdm::internal::callable_traits<Callable>::arity; 
  auto lam = [func, m](std::size_t i){ return fornfdm::Coordinate<N>(m, i).apply(func); }; 
  return Eigen::CwiseNullaryOp<decltype(lam), fornfdm::Vector>(s, 1, lam); 
}

auto discretize(std::shared_ptr<const fornfdm::Mesh> mesh, fornfdm::Scalar a)
{
  std::size_t s = mesh->sizesProduct(); 
  auto lam = [a](std::size_t i){ return a; }; 
  return Eigen::CwiseNullaryOp<decltype(lam), fornfdm::Vector>(s, 1, std::move(lam)); 
}

template<class Callable>
auto discretize(std::shared_ptr<const fornfdm::Mesh> mesh, const Callable& func)
{
  std::size_t s = mesh->sizesProduct(); 
  constexpr std::size_t N = fornfdm::internal::callable_traits<Callable>::arity; 
  auto lam = [func, m = std::move(mesh)](std::size_t i){ return fornfdm::Coordinate<N>(m.get(), i).apply(func); }; 
  return Eigen::CwiseNullaryOp<decltype(lam), fornfdm::Vector>(s, 1, std::move(lam)); 
}

auto linspaced(std::size_t n, fornfdm::Scalar x0, fornfdm::Scalar x1)
{
  auto lam = [d = static_cast<fornfdm::Scalar>(n)-1.0,x0,x1](std::size_t idx){ return (idx/d)*(x1-x0)+x0; };
  return Eigen::CwiseNullaryOp<decltype(lam), fornfdm::Vector>(n, 1, std::move(lam)); 
}

} // end namespace fornfdm 

#endif // Mesh.hpp 