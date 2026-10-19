# Unit and integration tests (GoogleTest).  All tests use the public host API
# only, so the same sources run against the CUDA and the CPU backend.
set(CRYOSPH_TESTS
  test_kernels
  test_spatial_hash
  test_geometry
  test_io
  test_hydrostatic
  test_conservation
  test_surface_tension
  test_transport_velocity
  test_determinism
  test_sessile_drop
  test_maneuver)

include(GoogleTest)
foreach(t IN LISTS CRYOSPH_TESTS)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE cryosph_core GTest::gtest_main)
  target_compile_definitions(${t} PRIVATE CRYOSPH_TEST_DATA_DIR="${CMAKE_CURRENT_BINARY_DIR}")
  gtest_discover_tests(${t} DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 900)
endforeach()
