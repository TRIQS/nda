// Copyright (c) 2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2024 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

#pragma once

/*

#include "./shared_array.hpp"

namespace nda {

template <typename ValueType, int Rank, typename LayoutPolicy = C_layout, char Algebra = 'A'>
class distributed_shared_array {
public:
    using local_array_t = shared_array<ValueType, Rank, LayoutPolicy, Algebra>;
    using layout_t = local_array_t::layout_t;
    using storage_t = local_array_t::storage_t;
    using shape_t = std::array<long, Rank>;

private:
    mpi::communicator world;
    mpi::shared_communicator shm;
    local_array_t local_array;
    shape_t global_shape;

    long num_nodes; // Number of nodes participating
    long my_node_index; // Rank of head process in head communicator

public:

    distributed_shared_array() = default;

    //Constructor takes global shape and communicator
    explicit distributed_shared_array(const shape_t &_global_shape, const mpi::communicator _world)
    : world(_world), shm(world.split_shared()), global_shape(_global_shape) {

        // form head communicator
        auto head = world.split(shm.rank() == 0 ? 0 : MPI_UNDEFINED);

        if (!head.is_null()) {
            num_nodes = head.size();
            my_node_index = head.rank();
        } else {
            my_node_index = -1;
        }
        mpi::broadcast(num_nodes, world);
        mpi::broadcast(my_node_index, shm);

        int total_rows = _global_shape[0];
        int base = total_rows / num_nodes;
        int remainder = total_rows % num_nodes;
        int local_rows = base + (my_node_index < remainder ? 1 : 0);

        shape_t local_shape = global_shape;
        local_shape[0] = local_rows; //use .resize() ?

        local_array = local_array_t(local_shape, shm);
    }

    explicit distributed_shared_array(const distributed_shared_array&) = default;

    distributed_shared_array(distributed_shared_array&&) = default;

    distributed_shared_array& operator=(const distributed_shared_array&) = default;

    distributed_shared_array& operator=(distributed_shared_array&&) = default;


    local_array_t &local() { return local_array; }
    const shape_t &shape() const { return global_shape; }

    void fence(bool global_sync = false) {
        if (!global_sync) {
            nda::fence(local_array);
        } else {
            world.barrier();
        }

    }


    template <typename Functor>
    void for_each_chunked(Functor &&f, long n_chunks, long rank) {
        nda::for_each_chunked(std::forward<Functor>(f), local_array, n_chunks, rank);
    }

    template <typename Functor>
    void for_each(Functor&& f) {
        auto &lay = local_array.indexmap();
        for (long i = 0; i < lay.size(); ++i) {
            f(local_array(nda::_linear_index_t{i}));
        }
    }

};



}

*/
