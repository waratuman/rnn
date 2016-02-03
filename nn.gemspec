Gem::Specification.new do |s|
  s.name = 'nn'
  s.version = '0.0.1'
  s.summary = 'Bindings for the libnn Library.'
  s.author = 'James Bracy'

  s.files = Dir.glob('ext/**/*.{c,h,rb}') +
            Dir.glob('lib/**/*.rb')
  
  s.extensions << 'ext/nn/extconf.rb'

  s.add_development_dependency 'rake-compiler'
end
