#pragma once

#define ARRAY_TYPE std::size_t

#define VP_TYPE double
#define WP_TYPE double

enum backend_impl { naive = 0, cpu_ifirst, cpu_kfirst, gpu };

enum class Data { ifirst, kfirst };
