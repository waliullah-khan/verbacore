const webpack = require('webpack');

module.exports = function override(config) {
  // Add the Node.js polyfills
  const fallback = {
    "buffer": require.resolve("buffer/"),
    "process": require.resolve("process/browser"),
    "util": require.resolve("util/"),
    "path": require.resolve("path-browserify"),
    "stream": require.resolve("stream-browserify"),
    "os": require.resolve("os-browserify/browser"),
    "fs": false,
    "crypto": false,
    "assert": false,
    "http": false,
    "https": false,
    "url": false,
    "zlib": false,
    "net": false,
    "tls": false,
    "child_process": false
  };

  config.resolve.fallback = {
    ...(config.resolve.fallback || {}),
    ...fallback
  };

  // Add webpack plugins for polyfills
  config.plugins = [
    ...config.plugins,
    new webpack.ProvidePlugin({
      Buffer: ['buffer', 'Buffer'],
      process: 'process/browser',
    }),
    new webpack.NormalModuleReplacementPlugin(/node:/, (resource) => {
      const mod = resource.request.replace(/^node:/, '');
      
      if (mod === 'buffer') {
        resource.request = 'buffer';
      } else if (mod === 'stream') {
        resource.request = 'stream-browserify';
      } else if (mod === 'util') {
        resource.request = 'util';
      } else if (mod === 'path') {
        resource.request = 'path-browserify';
      }
    }),
  ];

  // Tell webpack not to generate warnings about these modules
  config.ignoreWarnings = [
    ...config.ignoreWarnings || [],
    /Failed to parse source map/,
    /Critical dependency/
  ];

  return config;
}; 