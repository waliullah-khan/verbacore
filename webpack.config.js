const path = require('path');

module.exports = {
  resolve: {
    fallback: {
      "util": require.resolve("util/"),
      "path": require.resolve("path-browserify"),
      "stream": require.resolve("stream-browserify"),
      "buffer": require.resolve("buffer/"),
      "fs": false,
      "crypto": false,
      "os": false,
      "assert": false,
      "http": false,
      "https": false,
      "url": false,
      "zlib": false,
      "net": false,
      "tls": false,
      "child_process": false
    }
  }
}; 