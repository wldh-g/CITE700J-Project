import React from 'react';
import { Icon } from '@fluentui/react';
import './style.scss';

const { remote } = window.module.require('electron');

type Props = {
  onClick: (e: any) => any
  icon: string
  background?: string
  color?: string
}

type States = {
  hover: boolean
  push: boolean
}

class WindowButton extends React.Component<Props, States> {
  static defaultProps = {
    background: 'var(--over-color)',
    color: remote.nativeTheme.shouldUseDarkColors ? 'white' : 'black',
  }

  constructor(props) {
    super(props);

    this.state = { hover: false, push: false };

    // Bind functions
    this.hoverOn = this.hoverOn.bind(this);
    this.hoverOff = this.hoverOff.bind(this);
    this.pushOn = this.pushOn.bind(this);
    this.pushOff = this.pushOff.bind(this);
  }

  hoverOn() { this.setState({ hover: true }); }

  hoverOff() { this.setState({ hover: false, push: false }); }

  pushOn() { this.setState({ push: true, hover: true }); }

  pushOff() { this.setState({ push: false }); }

  render() {
    const {
      onClick, icon, background, color,
    } = this.props;
    const { hover, push } = this.state;
    const buttonStyle = {
      backgroundColor: hover ? background : 'transparent',
      color: hover ? color : 'var(--fore-color)',
      opacity: push ? 0.8 : 1,
    };
    return (
      <button
        styleName="window-button"
        type="button"
        style={buttonStyle}
        onMouseOver={this.hoverOn}
        onMouseOut={this.hoverOff}
        onFocus={this.hoverOn}
        onBlur={this.hoverOff}
        onMouseDown={this.pushOn}
        onMouseUp={this.pushOff}
        onClick={onClick}
      >
        <Icon iconName={icon} />
      </button>
    );
  }
}

export default WindowButton;
