#pragma once
/*
 *  Copyright (C) 2024  Brett Terpstra
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef COSC_4P80_ASSIGNMENT_2_LAYER_H
#define COSC_4P80_ASSIGNMENT_2_LAYER_H

#include <blt/std/types.h>
#include <assign2/initializers.h>
#include "blt/iterator/zip.h"
#include "blt/iterator/iterator.h"

namespace assign2
{
    class neuron_t
    {
        public:
            // empty neuron for loading from a stream
            explicit neuron_t(weight_view weights): weights(weights)
            {}
            
            // neuron with bias
            explicit neuron_t(weight_view weights, Scalar bias): bias(bias), weights(weights)
            {}
            
            template<typename ActFunc>
            Scalar activate(const Scalar* inputs, ActFunc func) const
            {
                auto sum = bias;
                for (auto [x, w] : blt::zip_iterator_container({inputs, inputs + weights.size()}, {weights.begin(), weights.end()}))
                    sum += x * w;
                return func.call(sum);
            }
            
            template<typename OStream>
            OStream& serialize(OStream& stream)
            {
                stream << bias;
                for (auto d : weights)
                    stream << d;
            }
            
            template<typename IStream>
            IStream& deserialize(IStream& stream)
            {
                for (auto& d : blt::iterate(weights).rev())
                    stream >> d;
                stream >> bias;
            }
        
        private:
            Scalar bias = 0;
            weight_view weights;
    };
    
    class layer_t
    {
        public:
            template<typename WeightFunc, typename BiasFunc>
            layer_t(const blt::i32 in, const blt::i32 out, WeightFunc w, BiasFunc b): in_size(in), out_size(out)
            {
                neurons.reserve(out_size);
                for (blt::i32 i = 0; i < out_size; i++)
                {
                    auto weight = weights.allocate_view(in_size);
                    for (auto& v : weight)
                        v = w(i);
                    neurons.push_back(neuron_t{weight, b(i)});
                }
            }
            
            template<typename ActFunction>
            std::vector<Scalar> call(const std::vector<Scalar>& in, ActFunction func = ActFunction{})
            {
                std::vector<Scalar> out;
                out.reserve(out_size);
#if BLT_DEBUG_LEVEL > 0
                if (in.size() != in_size)
                    throw std::runtime_exception("Input vector doesn't match expected input size!");
#endif
                for (auto& n : neurons)
                    out.push_back(n.activate(in.data(), func));
                return out;
            }
            
            template<typename OStream>
            OStream& serialize(OStream& stream)
            {
                for (auto d : neurons)
                    stream << d;
            }
            
            template<typename IStream>
            IStream& deserialize(IStream& stream)
            {
                for (auto& d : blt::iterate(neurons).rev())
                    stream >> d;
            }
            
            [[nodiscard]] inline blt::i32 get_in_size() const
            {
                return in_size;
            }
            
            [[nodiscard]] inline blt::i32 get_out_size() const
            {
                return out_size;
            }
        private:
            const blt::i32 in_size, out_size;
            weight_t weights;
            std::vector<neuron_t> neurons;
    };
}

#endif //COSC_4P80_ASSIGNMENT_2_LAYER_H
