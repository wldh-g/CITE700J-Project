const path = require('path');
const fs = require('fs');
const webpack = require('webpack');

const packageInfo = JSON.parse(fs.readFileSync('./package.json').toString());
const port = packageInfo.port;

module.exports = {
  entry: {
    main: path.resolve(__dirname, 'src', 'components', 'main.tsx'),
    live_c: path.resolve(__dirname, 'src', 'components', 'viewOnC.tsx'),
    //live_cuda: path.resolve(__dirname, 'src', 'components', 'viewOnCUDA.tsx'),
    live_simd: path.resolve(__dirname, 'src', 'components', 'viewOnSIMD.tsx'),
  },
  mode: 'development',
  module: {
    rules: [
      {
        test: /\.(ts|tsx)$/,
        exclude: /node_modules/,
        loader: [
          'babel-loader',
          {
            loader: 'ts-loader',
            options: {
              transpileOnly: true,
            },
          },
        ],
      },
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        loader: 'babel-loader',
      },
      {
        test: /\.css$/,
        use: [
          'style-loader',
          {
            loader: 'css-loader',
            query: {
              modules: {
                localIdentName: '[local]',
              },
              importLoaders: 1,
              sourceMap: true,
            },
          },
        ],
      },
      {
        test: /\.scss$/,
        use: [
          'style-loader',
          {
            loader: 'css-loader',
            query: {
              modules: {
                localIdentName: '[local]',
              },
              importLoaders: 1,
              sourceMap: true,
            },
          },
          'sass-loader',
        ],
      },
    ],
  },
  resolve: {
    extensions: ['.js', '.jsx', '.ts', '.tsx'],
    alias: {
      'react-dom': '@hot-loader/react-dom',
      components: path.resolve(__dirname, 'src', 'components'),
    },
  },
  output: {
    path: path.resolve(__dirname, 'public'),
    publicPath: '/',
    filename: path.join('dist', '[name].js'),
  },
  devServer: {
    contentBase: path.join(__dirname, 'public'),
    port,
    publicPath: `http://localhost:${port}/`,
    hotOnly: true,
  },
  devtool: 'source-map',
  plugins: [new webpack.HotModuleReplacementPlugin()],
};
