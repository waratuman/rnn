require 'rake'
require 'rake/extensiontask'

spec = Gem::Specification.load('nn.gemspec')

Gem::PackageTask.new(spec)

Rake::ExtensionTask.new do |ext|
  ext.gem_spec = spec
  ext.name = 'nn'
  ext.ext_dir = 'ext/nn'
  ext.lib_dir = 'lib/nn'
end

task :console do
  $:.unshift 'lib'
  require 'nn'
  require 'irb'
  require 'irb/completion'
  ARGV.clear
  IRB.start
end