const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

// Get the project root
const projectRoot = __dirname;
// Get the workspace root
const workspaceRoot = path.resolve(projectRoot, '../..');

const config = getDefaultConfig(projectRoot);

// 1. Watch all files in the workspace
config.watchFolders = [workspaceRoot];

// 2. Let Metro know where to resolve packages and in what order
config.resolver.nodeModulesPaths = [
  path.resolve(projectRoot, 'node_modules'),
  path.resolve(workspaceRoot, 'node_modules'),
];

// 3. Force Metro to resolve symlinks, so it can find the linked package
config.resolver.unstable_enableSymlinks = true;
config.resolver.unstable_enablePackageExports = true;

// 4. Add a direct mapping to the package
config.resolver.extraNodeModules = {
  'cactus-react-native': path.resolve(workspaceRoot, 'cactus-react'),
};

module.exports = config; 