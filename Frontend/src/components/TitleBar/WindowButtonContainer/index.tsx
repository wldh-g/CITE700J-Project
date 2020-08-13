import React from 'react';
import './style.scss';

interface Props {
  // eslint-disable-next-line react/require-default-props
  children?: React.ReactNode,
}

// eslint-disable-next-line import/no-anonymous-default-export
export default (props: Props): React.ReactElement<any> => {
  const { children } = props;
  return (<div styleName="window-button-container">{children}</div>);
};
