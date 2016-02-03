require 'mkmf'

$CFLAGS << ' -g -O2 -Wall -Wextra -Wpedantic -Wstrict-overflow -fno-strict-aliasing'

LIB_DIRS = ['/usr/local/lib', RbConfig::CONFIG['libdir'], '/usr/lib']
HEADER_DIRS = ['/usr/local/include', RbConfig::CONFIG['includedir'], '/usr/include']

dir_config('nn/nn', HEADER_DIRS, LIB_DIRS)

if !have_library('nn')
  abort('Could not find the NN library (libnn)')
end

create_makefile('nn/nn')

