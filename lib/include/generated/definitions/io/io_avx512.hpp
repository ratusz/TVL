/*==========================================================================*
 * This file is part of the TVL - a template SIMD library.                  *
 *                                                                          *
 * Copyright 2022 TVL-Team, Database Research Group TU Dresden              *
 *                                                                          *
 * Licensed under the Apache License, Version 2.0 (the "License");          *
 * you may not use this file except in compliance with the License.         *
 * You may obtain a copy of the License at                                  *
 *                                                                          *
 *     http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                          *
 * Unless required by applicable law or agreed to in writing, software      *
 * distributed under the License is distributed on an "AS IS" BASIS,        *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 * See the License for the specific language governing permissions and      *
 * limitations under the License.                                           *
 *==========================================================================*/
/*
 * \file /home/runner/work/TVLGen/TVLGen/lib/include/generated/definitions/io/io_avx512.hpp
 * \date 2022-08-25
 * \brief Input/Output primitives.
 * \note
 * Git-Local Url : /home/runner/work/TVLGen/TVLGen/generator
 * Git-Remote Url: git@github.com:db-tu-dresden/TVLGen.git
 * Git-Branch    : main
 * Git-Commit    : 09e790d (09e790d9cc40c4de3fbe683afe9241a803f25915)
 * Submodule(s):
 *   Git-Local Url : primitive_data
 *   Git-Remote Url: git@github.com:db-tu-dresden/TVLPrimitiveData.git
 *   Git-Branch    : main
 *   Git-Commit    : be13979 (be139799c677e3ddb66ae930c197c81312c29456)
 *
 */
#ifndef TUD_D2RG_TVL_HOME_RUNNER_WORK_TVLGEN_TVLGEN_LIB_INCLUDE_GENERATED_DEFINITIONS_IO_IO_AVX512_HPP
#define TUD_D2RG_TVL_HOME_RUNNER_WORK_TVLGEN_TVLGEN_LIB_INCLUDE_GENERATED_DEFINITIONS_IO_IO_AVX512_HPP

#include "../../declarations/io.hpp"
namespace tvl {
   namespace functors {
      /**
       * @brief: Template specialization of implementation for "to_ostream".
       * @details:
       * Target Extension: avx512.
       *        Data Type: int64_t
       *  Extension Flags: ['avx512f']
       */
      template<ImplementationDegreeOfFreedom Idof>
         struct to_ostream<simd<int64_t, avx512>, Idof> {
            using Vec = simd<int64_t, avx512>;
            static constexpr bool native_supported() {
               return true;
            }
            [[nodiscard]] 
            TVL_FORCE_INLINE 
            static std::ostream & apply(
                std::ostream & out, typename Vec::register_type const data
            ) {
               for(auto idx = 0; idx < 8; ++idx ) {
   out << data[7-idx] << "|";
}
out << "\n";
return out;
            }
         };
   } // end of namespace functors for template specialization of to_ostream for avx512 using int64_t.
} // end of namespace tvl
#endif //TUD_D2RG_TVL_HOME_RUNNER_WORK_TVLGEN_TVLGEN_LIB_INCLUDE_GENERATED_DEFINITIONS_IO_IO_AVX512_HPP